"""
Conv Encoder — 8-layer conv backbone + full GeoLIP classification pipeline.

ConvEncoder:        Implements Input stage. 8-layer conv → S^(d-1).
GeoLIPConvEncoder:  ConvEncoder → MagnitudeFlow → ConstellationObserver → task head.

The observer observes. The task head and CE loss live HERE.

Usage:
    from geolip_core.encoder.encoder_conv import ConvEncoder, GeoLIPConvEncoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.core.util import make_activation, param_count, model_summary
from geolip_core.pipeline.observer import Input
from geolip_core.core.associate.constellation import ConstellationObserver
from geolip_core.core.curate.patchwork import MagnitudeFlow
from geolip_core.core.distinguish.losses import ce_loss_paired, observer_loss


# ══════════════════════════════════════════════════════════════════
# CONV ENCODER — Input stage: pixels → S^(d-1)
# ══════════════════════════════════════════════════════════════════

class ConvEncoder(Input):
    """8-layer conv → D-dim projection. Implements Input interface.

    Architecture: 4 blocks of (conv-BN-GELU, conv-BN-GELU, MaxPool)
    Channels: 64 → 128 → 256 → 384

    encode() returns unnormalized features.
    forward() (inherited from Input) returns (emb on S^(d-1), magnitude).

    Args:
        output_dim: embedding dimension (default 256)
    """

    def __init__(self, output_dim=256, **kwargs):
        super().__init__(**kwargs)
        self._dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384), nn.GELU(),
            nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(384, output_dim),
            nn.LayerNorm(output_dim),
        )

    @property
    def dim(self):
        return self._dim

    def encode(self, x):
        """x: (B, 3, H, W) → (B, dim) unnormalized features."""
        return self.proj(self.features(x))


# ══════════════════════════════════════════════════════════════════
# GEOLIP CONV ENCODER — full pipeline with observer framework
# ══════════════════════════════════════════════════════════════════

class GeoLIPConvEncoder(nn.Module):
    """ConvEncoder wired to ConstellationObserver + classification head.

    ConvEncoder → L2 normalize → MagnitudeFlow → ConstellationObserver → task head.

    The observer handles observation and self-organization. The task head
    and CE loss are owned HERE — classification is this class's job.

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
        mag_hidden=64,
        mag_heads=4,
        mag_layers=2,
        mag_min=0.1,
        mag_max=5.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.cv_target = cv_target
        self.infonce_temp = infonce_temp
        self.assign_temp = assign_temp
        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and not k.startswith('_')}

        # Feature extraction
        self.encoder = ConvEncoder(output_dim)

        # Magnitude prediction
        self.mag_flow = MagnitudeFlow(
            dim=output_dim, n_anchors=n_anchors,
            hidden_dim=mag_hidden, n_heads=mag_heads, n_layers=mag_layers,
            mag_min=mag_min, mag_max=mag_max, n_comp=n_comp,
        )

        # Observer — holds the constellation observation pipeline
        self.observer = ConstellationObserver(
            dim=output_dim, n_anchors=n_anchors,
            n_comp=n_comp, d_comp=d_comp,
            anchor_drop=anchor_drop, anchor_init='repulsion',
            activation=activation, assign_temp=assign_temp,
        )

        # Task head — classification is THIS class's job
        self.task_head = nn.Sequential(
            nn.Linear(self.observer.feature_dim, self.observer.patchwork.output_dim),
            make_activation(activation),
            nn.LayerNorm(self.observer.patchwork.output_dim),
            nn.Dropout(0.1),
            nn.Linear(self.observer.patchwork.output_dim, num_classes),
        )

        self._init_encoder_weights()

    @property
    def constellation_observer(self):
        """Convenience accessor."""
        return self.observer

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
        """Pixels → S^(d-1) + per-anchor magnitude."""
        emb, raw_mag = self.encoder(x)  # Input.forward() returns (emb, magnitude)
        obs = self.constellation_observer
        anchors_n = F.normalize(obs.constellation.anchors, dim=-1)
        tri = emb @ anchors_n.T
        mag, mag_comp = self.mag_flow(emb, tri, raw_mag)
        return emb, mag, mag_comp

    def forward_paired(self, v1, v2):
        """Training: two views → observation + logits."""
        emb1, mag1, mc1 = self._encode(v1)
        emb2, mag2, mc2 = self._encode(v2)
        out = self.observer.observe_paired(emb1, emb2, mag1=mag1, mag2=mag2)
        out['logits'] = self.task_head(out['features1'])
        out['logits_aug'] = self.task_head(out['features2'])
        out['mag_comp1'] = mc1
        out['mag_comp2'] = mc2
        return out

    def forward(self, x):
        """Eval: single view → observation + logits."""
        emb, mag, mag_comp = self._encode(x)
        out = self.observer.observe(emb, mag=mag)
        out['logits'] = self.task_head(out['features'])
        out['mag_comp'] = mag_comp
        return out

    def compute_loss(self, output, targets, w_ce=1.0, **loss_kwargs):
        """Observer loss + CE classification loss.

        Returns: (total_loss, loss_dict)
        """
        obs = self.observer

        # Observer self-organization (keys are flat — no namespace prefix)
        obs_loss, ld = observer_loss(
            output,
            anchors=obs.constellation.anchors,
            targets=targets,
            infonce_temp=self.infonce_temp,
            assign_temp=self.assign_temp,
            cv_target=self.cv_target,
            **loss_kwargs,
        )

        # Classification — THIS class's concern
        l_ce, acc = ce_loss_paired(output['logits'], output['logits_aug'], targets)
        ld['ce'] = l_ce
        ld['acc'] = acc

        # Total
        loss = w_ce * l_ce + obs_loss
        ld['loss_task'] = l_ce.item()
        ld['loss_observer'] = obs_loss.item()
        ld['total'] = loss
        return loss, ld

    def get_anchor_param_ids(self):
        """Param ids that should have weight_decay=0."""
        obs = self.constellation_observer
        ids = set(id(p) for p in obs.constellation.parameters())
        for relay in self.mag_flow.relays:
            ids.add(id(relay.anchors))
        return ids

    def make_optimizer(self, lr=3e-4, weight_decay=0.05):
        """Build AdamW with anchor exclusion from weight decay."""
        anchor_ids = self.get_anchor_param_ids()
        decay = [p for p in self.parameters() if id(p) not in anchor_ids]
        nodecay = [p for p in self.parameters() if id(p) in anchor_ids]
        return torch.optim.AdamW([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ], lr=lr)

    def summary(self):
        obs = self.constellation_observer
        print("GeoLIPConvEncoder Summary")
        print("=" * 50)
        param_count(self.encoder, "encoder")
        param_count(self.mag_flow, "mag_flow")
        param_count(obs.constellation, "constellation")
        param_count(obs.patchwork, "patchwork")
        param_count(obs.bridge, "bridge")
        param_count(self.task_head, "task_head")
        print("-" * 50)
        total = model_summary(self)
        print(f"\n  Config: {self.config}")
        return total