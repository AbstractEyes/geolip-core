"""
Constellation Encoder — full GeoLIP image pipeline.

InternalConstellationCore: Three-domain head (external + geometric + internal).
GeoLIPImageEncoder: ConvEncoder → S^(d-1) → MagnitudeFlow → Core.

Usage:
    from encoder.encoder_constellation import GeoLIPImageEncoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.activation import make_activation
from ..core.constellation import Constellation
from ..core.patchwork import Patchwork, MagnitudeFlow
from ..core.core import param_count, model_summary
from ..core.losses import (
    cv_loss, spread_loss, attraction_loss,
    nce_loss, ce_loss_paired, bridge_loss_paired,
    assign_bce_loss, assign_nce_loss, knn_accuracy,
)
from .encoder_conv import ConvEncoder


# ══════════════════════════════════════════════════════════════════
# INTERNAL CONSTELLATION CORE — three-domain head
# ══════════════════════════════════════════════════════════════════

class InternalConstellationCore(nn.Module):
    """Constellation with independent internal + external objectives.

    Three domains:
      EXTERNAL:   CE + embedding NCE → task_head, patchwork, encoder
      GEOMETRIC:  patchwork NCE + bridge → patchwork, encoder, anchors
      INTERNAL:   assign + tri NCE + attract + CV + spread → anchors, encoder

    Args:
        num_classes: classification targets
        dim: embedding dimension
        n_anchors: anchors on S^(dim-1)
        n_comp: patchwork compartments
        d_comp: hidden dim per compartment
        anchor_drop: training anchor dropout
        activation: activation function name
        cv_target: target CV for geometric loss
        infonce_temp: embedding NCE temperature
        assign_temp: assignment temperature
        assign_sharpness: BCE target sharpness
    """

    def __init__(
        self,
        num_classes=100,
        dim=256,
        n_anchors=128,
        n_comp=8,
        d_comp=64,
        anchor_drop=0.15,
        activation='squared_relu',
        cv_target=0.22,
        infonce_temp=0.07,
        assign_temp=0.1,
        assign_sharpness=5.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.n_anchors = n_anchors
        self.cv_target = cv_target
        self.infonce_temp = infonce_temp
        self.assign_temp = assign_temp
        self.assign_sharpness = assign_sharpness

        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and not k.startswith('_')}

        self.constellation = Constellation(n_anchors, dim, anchor_drop)
        self.patchwork = Patchwork(n_anchors, n_comp, d_comp, activation)
        pw_dim = self.patchwork.output_dim

        self.bridge = nn.Sequential(nn.Linear(pw_dim, n_anchors))

        total_feat = n_anchors + pw_dim + dim
        self.task_head = nn.Sequential(
            nn.Linear(total_feat, pw_dim),
            make_activation(activation),
            nn.LayerNorm(pw_dim),
            nn.Dropout(0.1),
            nn.Linear(pw_dim, num_classes),
        )

        self.register_buffer('anchor_classes', torch.zeros(n_anchors, dtype=torch.long))
        self.register_buffer('class_centroids', torch.zeros(num_classes, dim))

    def _triangulate(self, emb):
        anchors_n = F.normalize(self.constellation.anchors, dim=-1)
        cos = emb @ anchors_n.T
        tri = 1.0 - cos
        _, nearest = cos.max(dim=-1)
        soft_assign = F.softmax(cos / self.assign_temp, dim=-1)
        return cos, tri, nearest, soft_assign

    def forward_paired(self, emb1, emb2, mag1=None, mag2=None):
        """Paired forward for training. Returns dict with all intermediates."""
        cos1, tri1, nearest1, assign1 = self._triangulate(emb1)
        cos2, tri2, nearest2, assign2 = self._triangulate(emb2)

        tri1_w = tri1 * mag1 if mag1 is not None else tri1
        tri2_w = tri2 * mag2 if mag2 is not None else tri2

        pw1 = self.patchwork(tri1_w)
        pw2 = self.patchwork(tri2_w)
        bridge1 = self.bridge(pw1)
        bridge2 = self.bridge(pw2)

        feat1 = torch.cat([assign1, pw1, emb1], dim=-1)
        feat2 = torch.cat([assign2, pw2, emb2], dim=-1)
        logits1 = self.task_head(feat1)
        logits2 = self.task_head(feat2)

        return {
            'embedding': emb1, 'embedding_aug': emb2,
            'mag1': mag1, 'mag2': mag2,
            'cos1': cos1, 'cos2': cos2,
            'tri1': tri1, 'tri2': tri2,
            'nearest': nearest1,
            'assign1': assign1, 'assign2': assign2,
            'patchwork1': pw1, 'patchwork1_aug': pw2,
            'bridge1': bridge1, 'bridge2': bridge2,
            'logits': logits1, 'logits_aug': logits2,
        }

    def forward(self, emb, mag=None):
        """Single view for eval."""
        out = self.forward_paired(emb, emb, mag, mag)
        return {
            'logits': out['logits'],
            'embedding': emb,
            'magnitude': mag,
            'triangulation': out['tri1'],
            'cos_to_anchors': out['cos1'],
            'nearest': out['nearest'],
            'assignment': out['assign1'],
            'patchwork': out['patchwork1'],
        }

    def compute_loss(self, output, targets,
                     w_ce=1.0, w_nce_emb=0.5,
                     w_nce_pw=1.0, w_bridge=1.0,
                     w_assign=0.5, w_assign_nce=0.25,
                     w_nce_tri=0.5, w_attract=0.25,
                     w_cv=0.01, w_spread=0.01,
                     cv_batched=True):
        """Three-domain cooperative loss. Returns (total_loss, loss_dict)."""
        ld = {}
        emb1, emb2 = output['embedding'], output['embedding_aug']

        l_ce, acc = ce_loss_paired(output['logits'], output['logits_aug'], targets)
        ld['ce'], ld['acc'] = l_ce, acc
        l_nce_emb, nce_emb_acc = nce_loss(emb1, emb2, self.infonce_temp, normalize=False)
        ld['nce_emb'], ld['nce_emb_acc'] = l_nce_emb, nce_emb_acc

        l_nce_pw, nce_pw_acc = nce_loss(output['patchwork1'], output['patchwork1_aug'], self.assign_temp, normalize=True)
        ld['nce_pw'], ld['nce_pw_acc'] = l_nce_pw, nce_pw_acc
        l_bridge, bridge_acc = bridge_loss_paired(output['bridge1'], output['bridge2'], output['assign1'], output['assign2'])
        ld['bridge'], ld['bridge_acc'] = l_bridge, bridge_acc

        l_assign, assign_ent = assign_bce_loss(output['assign1'], output['cos1'])
        ld['assign'], ld['assign_entropy'] = l_assign, assign_ent
        l_assign_nce, assign_nce_acc = assign_nce_loss(output['assign1'], output['assign2'], self.assign_temp)
        ld['assign_nce'], ld['assign_nce_acc'] = l_assign_nce, assign_nce_acc
        l_nce_tri, nce_tri_acc = nce_loss(output['tri1'], output['tri2'], 0.1, normalize=True)
        ld['nce_tri'], ld['nce_tri_acc'] = l_nce_tri, nce_tri_acc
        l_attract, nearest_cos = attraction_loss(output['cos1'])
        ld['attract'], ld['nearest_cos'] = l_attract, nearest_cos
        l_cv = cv_loss(emb1, target=self.cv_target, batched=cv_batched)
        ld['cv'] = l_cv
        l_spread = spread_loss(self.constellation.anchors)
        ld['spread'] = l_spread

        ld['knn_acc'] = knn_accuracy(emb1, targets)

        loss_external = w_ce * l_ce + w_nce_emb * l_nce_emb
        loss_geometric = w_nce_pw * l_nce_pw + w_bridge * l_bridge
        loss_internal = (w_assign * l_assign + w_assign_nce * l_assign_nce
                         + w_nce_tri * l_nce_tri + w_attract * l_attract
                         + w_cv * l_cv + w_spread * l_spread)
        loss = loss_external + loss_geometric + loss_internal

        ld['loss_external'] = loss_external.item()
        ld['loss_geometric'] = loss_geometric.item()
        ld['loss_internal'] = loss_internal.item()
        ld['t_ce'] = l_ce.item()
        ld['t_nce_emb'] = l_nce_emb.item()
        ld['t_nce_pw'] = l_nce_pw.item()
        ld['t_bridge'] = l_bridge.item()
        ld['t_assign'] = l_assign.item()
        ld['t_assign_nce'] = l_assign_nce.item()
        ld['t_nce_tri'] = l_nce_tri.item()
        ld['t_attract'] = l_attract.item()
        ld['total'] = loss
        return loss, ld


# ══════════════════════════════════════════════════════════════════
# GEOLIP IMAGE ENCODER — full pipeline
# ══════════════════════════════════════════════════════════════════

class GeoLIPImageEncoder(nn.Module):
    """Complete GeoLIP model: ConvEncoder → S^(d-1) → MagnitudeFlow → Core.

    Args:
        num_classes: classification targets
        output_dim: embedding dimension on S^(d-1)
        n_anchors: constellation anchors
        n_comp: patchwork compartments
        d_comp: per-compartment hidden dim
        anchor_drop: training anchor dropout
        activation: activation function name
        cv_target: CV loss target
        infonce_temp: embedding NCE temperature
        assign_temp: assignment temperature
        assign_sharpness: BCE sharpness
        mag_hidden: magnitude relay hidden dim
        mag_heads: unused (API compat)
        mag_layers: relay layers in MagnitudeFlow
        mag_min: minimum magnitude
        mag_max: maximum magnitude
    """

    def __init__(
        self,
        num_classes=100,
        output_dim=384,
        n_anchors=512,
        n_comp=8,
        d_comp=64,
        anchor_drop=0.15,
        activation='squared_relu',
        cv_target=0.22,
        infonce_temp=0.07,
        assign_temp=0.1,
        assign_sharpness=5.0,
        mag_hidden=64,
        mag_heads=4,
        mag_layers=2,
        mag_min=0.1,
        mag_max=5.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and not k.startswith('_')}

        self.encoder = ConvEncoder(output_dim)
        self.mag_flow = MagnitudeFlow(
            dim=output_dim, n_anchors=n_anchors,
            hidden_dim=mag_hidden, n_heads=mag_heads, n_layers=mag_layers,
            mag_min=mag_min, mag_max=mag_max, n_comp=n_comp,
        )
        self.core = InternalConstellationCore(
            num_classes=num_classes, dim=output_dim,
            n_anchors=n_anchors, n_comp=n_comp, d_comp=d_comp,
            anchor_drop=anchor_drop, activation=activation,
            cv_target=cv_target, infonce_temp=infonce_temp,
            assign_temp=assign_temp, assign_sharpness=assign_sharpness,
        )
        self._init_encoder_weights()

    def _init_encoder_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _encode(self, x):
        feat = self.encoder(x)
        raw_mag = feat.norm(dim=-1, keepdim=True)
        emb = F.normalize(feat, dim=-1)
        anchors_n = F.normalize(self.core.constellation.anchors, dim=-1)
        tri = emb @ anchors_n.T
        mag, mag_comp = self.mag_flow(emb, tri, raw_mag)
        return emb, mag, mag_comp

    def forward_paired(self, v1, v2):
        """Training: two views → full pipeline."""
        emb1, mag1, mc1 = self._encode(v1)
        emb2, mag2, mc2 = self._encode(v2)
        out = self.core.forward_paired(emb1, emb2, mag1, mag2)
        out['mag_comp1'] = mc1
        out['mag_comp2'] = mc2
        return out

    def forward(self, x):
        """Eval: single view → classify."""
        emb, mag, mag_comp = self._encode(x)
        out = self.core(emb, mag)
        out['mag_comp'] = mag_comp
        return out

    def compute_loss(self, output, targets, **kwargs):
        return self.core.compute_loss(output, targets, **kwargs)

    def get_anchor_param_ids(self):
        ids = set(id(p) for p in self.core.constellation.parameters())
        for relay in self.mag_flow.relays:
            ids.add(id(relay.anchors))
        return ids

    def make_optimizer(self, lr=3e-4, weight_decay=0.05):
        """Build AdamW with anchor exclusion from weight decay.

        For pure Adam (no weight decay), use:
            torch.optim.Adam(model.parameters(), lr=lr)
        """
        anchor_ids = self.get_anchor_param_ids()
        decay = [p for p in self.parameters() if id(p) not in anchor_ids]
        nodecay = [p for p in self.parameters() if id(p) in anchor_ids]
        return torch.optim.AdamW([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ], lr=lr)

    def summary(self):
        print("GeoLIPImageEncoder Summary")
        print("=" * 50)
        param_count(self.encoder, "encoder")
        param_count(self.mag_flow, "mag_flow")
        param_count(self.core.constellation, "constellation")
        param_count(self.core.patchwork, "patchwork")
        param_count(self.core.bridge, "bridge")
        param_count(self.core.task_head, "task_head")
        print("-" * 50)
        total = model_summary(self)
        print(f"\n  Config: {self.config}")
        return total