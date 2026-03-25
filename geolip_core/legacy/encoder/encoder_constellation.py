"""
Constellation Encoder — Full-Stack GeoLIP Composition
========================================================
GeoLIP tower with constellation stages attached as named components.

ConstellationEncoder:  emb on S^(d-1) → mag → observe → distinguish
ClassificationHead:    Distinction stage for supervised tasks
GeoLIPEncoder:         Any Input + ConstellationEncoder composed

Usage:
    from geolip_core.encoder import ConstellationEncoder, GeoLIPEncoder, ConvEncoder

    # Full stack — accepts embeddings from any source
    enc = ConstellationEncoder(dim=384, n_anchors=512, num_classes=100)
    out = enc.forward_paired(emb1, emb2, raw_mag1, raw_mag2)
    loss, ld = enc.compute_loss(out, targets)

    # Composed with any Input stage
    model = GeoLIPEncoder(ConvEncoder(384), dim=384, num_classes=100)
    out = model.forward_paired(view1, view2)
    loss, ld = model.compute_loss(out, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.observer import GeoLIP, Distinction
from ..core.constellation import ConstellationObserver
from ..core.patchwork import MagnitudeFlow
from ..core.activation import make_activation
from ..core.core import param_count, model_summary
from ..core.losses import ce_loss_paired, observer_loss


# ══════════════════════════════════════════════════════════════════
# CLASSIFICATION HEAD — Distinction stage
# ══════════════════════════════════════════════════════════════════

class ClassificationHead(Distinction):
    """Distinction stage for supervised classification."""

    def __init__(self, feature_dim, num_classes, hidden_dim=None,
                 activation='squared_relu', dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if hidden_dim is None:
            hidden_dim = feature_dim
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            make_activation(activation),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def distinguish(self, features, **context):
        return self.head(features)


# ══════════════════════════════════════════════════════════════════
# CONSTELLATION ENCODER — GeoLIP tower
# ══════════════════════════════════════════════════════════════════

class ConstellationEncoder(GeoLIP):
    """Full geometric pipeline as a GeoLIP tower.

    Attaches constellation stages as named components:
        self['mag_flow']  — MagnitudeFlow (relay-stack confidence)
        self['observer']  — ConstellationObserver (association + curation)
        self['task_head'] — ClassificationHead (optional)

    forward() defines the canonical pipeline.

    Args:
        dim: embedding dimension on S^(dim-1)
        n_anchors: constellation anchors
        n_comp: patchwork compartments
        d_comp: per-compartment hidden dim
        num_classes: classification targets (0 = observer only)
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
        dim=384,
        n_anchors=512,
        n_comp=8,
        d_comp=64,
        num_classes=100,
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
        super().__init__('constellation_encoder', dim, strict=False)
        self.cv_target = cv_target
        self.infonce_temp = infonce_temp
        self.assign_temp = assign_temp
        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and not k.startswith('_')}

        # Attach stages as named components
        self.attach('mag_flow', MagnitudeFlow(
            dim=dim, n_anchors=n_anchors,
            hidden_dim=mag_hidden, n_heads=mag_heads, n_layers=mag_layers,
            mag_min=mag_min, mag_max=mag_max, n_comp=n_comp,
        ))

        self.attach('observer', ConstellationObserver(
            dim=dim, n_anchors=n_anchors,
            n_comp=n_comp, d_comp=d_comp,
            anchor_drop=anchor_drop, assign_temp=assign_temp,
            activation=activation,
        ))

        if num_classes > 0:
            obs = self['observer']
            self.attach('task_head', ClassificationHead(
                feature_dim=obs.feature_dim,
                num_classes=num_classes,
                hidden_dim=obs.patchwork.output_dim,
                activation=activation,
            ))

        # Buffers for push compatibility
        self.register_buffer('anchor_classes',
                             torch.zeros(n_anchors, dtype=torch.long))
        self.register_buffer('class_centroids',
                             torch.zeros(max(num_classes, 1), dim))

    # ── Accessors ──

    @property
    def constellation(self):
        return self['observer'].constellation

    @property
    def patchwork(self):
        return self['observer'].patchwork

    @property
    def mag_flow(self):
        return self['mag_flow']

    # ── Magnitude ──

    def _compute_magnitude(self, emb, raw_magnitude):
        mag_flow = self['mag_flow']
        anchors_n = F.normalize(self.constellation.anchors, dim=-1)
        tri = emb @ anchors_n.T
        return mag_flow(emb, tri, raw_magnitude)

    # ── Forward: the pipeline ──

    def forward(self, emb, raw_magnitude=None):
        """Single view: emb on S^(d-1) → observation → distinction."""
        if raw_magnitude is None:
            raw_magnitude = torch.ones(emb.shape[0], 1, device=emb.device, dtype=emb.dtype)

        mag, mag_comp = self._compute_magnitude(emb, raw_magnitude)
        out = self['observer'].observe(emb, mag=mag)
        out['mag_comp'] = mag_comp

        if self.has('task_head'):
            out['logits'] = self['task_head'](out['features'])

        return out

    def forward_paired(self, emb1, emb2, raw_mag1=None, raw_mag2=None):
        """Two views: paired observation → distinction."""
        ones = lambda e: torch.ones(e.shape[0], 1, device=e.device, dtype=e.dtype)
        if raw_mag1 is None: raw_mag1 = ones(emb1)
        if raw_mag2 is None: raw_mag2 = ones(emb2)

        mag1, mc1 = self._compute_magnitude(emb1, raw_mag1)
        mag2, mc2 = self._compute_magnitude(emb2, raw_mag2)

        out = self['observer'].observe_paired(emb1, emb2, mag1=mag1, mag2=mag2)
        out['mag_comp1'] = mc1
        out['mag_comp2'] = mc2

        if self.has('task_head'):
            out['logits'] = self['task_head'](out['features1'])
            out['logits_aug'] = self['task_head'](out['features2'])

        return out

    # ── Loss ──

    def compute_loss(self, output, targets, w_ce=1.0, **loss_kwargs):
        """Observer self-organization + optional classification."""
        obs_loss, ld = observer_loss(
            output,
            anchors=self.constellation.anchors,
            targets=targets,
            infonce_temp=self.infonce_temp,
            assign_temp=self.assign_temp,
            cv_target=self.cv_target,
            **loss_kwargs,
        )

        if self.has('task_head') and 'logits' in output:
            l_ce, acc = ce_loss_paired(output['logits'], output['logits_aug'], targets)
            ld['ce'] = l_ce
            ld['acc'] = acc
            loss = w_ce * l_ce + obs_loss
            ld['loss_task'] = l_ce.item()
        else:
            loss = obs_loss

        ld['loss_observer'] = obs_loss.item()
        ld['total'] = loss
        return loss, ld

    # ── Anchor management ──

    def get_anchor_param_ids(self):
        """Param ids that must have weight_decay=0."""
        ids = set(id(p) for p in self.constellation.parameters())
        for relay in self['mag_flow'].relays:
            ids.add(id(relay.anchors))
        return ids

    def make_optimizer(self, lr=3e-4, weight_decay=0.05, extra_params=None):
        """AdamW with anchor exclusion from weight decay."""
        anchor_ids = self.get_anchor_param_ids()
        all_params = list(self.parameters())
        if extra_params is not None:
            all_params.extend(extra_params)
        decay = [p for p in all_params if id(p) not in anchor_ids]
        nodecay = [p for p in all_params if id(p) in anchor_ids]
        return torch.optim.AdamW([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ], lr=lr)

    def summary(self):
        print(f"ConstellationEncoder '{self.name}'")
        print("=" * 50)
        for name in self.components:
            param_count(self[name], name)
        print("-" * 50)
        return model_summary(self)


# ══════════════════════════════════════════════════════════════════
# GEOLIP ENCODER — Input + ConstellationEncoder
# ══════════════════════════════════════════════════════════════════

class GeoLIPEncoder(nn.Module):
    """Any Input stage + ConstellationEncoder.

    Attaches the Input on the ConstellationEncoder so all parameters
    are reachable through a single module tree.

    Args:
        input_stage: Input implementation (ConvEncoder, etc.)
        **kwargs: forwarded to ConstellationEncoder
    """

    def __init__(self, input_stage, **kwargs):
        super().__init__()
        dim = kwargs.pop('dim', input_stage.dim)
        self.enc = ConstellationEncoder(dim=dim, **kwargs)
        self.enc.attach('input', input_stage)
        self._init_input_weights()

    @property
    def input_stage(self):
        return self.enc['input']

    @property
    def config(self):
        return self.enc.config

    def _init_input_weights(self):
        for m in self.input_stage.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        emb, raw_mag = self.input_stage(x)
        return self.enc(emb, raw_mag)

    def forward_paired(self, v1, v2):
        emb1, mag1 = self.input_stage(v1)
        emb2, mag2 = self.input_stage(v2)
        return self.enc.forward_paired(emb1, emb2, mag1, mag2)

    def compute_loss(self, output, targets, **kwargs):
        return self.enc.compute_loss(output, targets, **kwargs)

    def make_optimizer(self, lr=3e-4, weight_decay=0.05):
        return self.enc.make_optimizer(
            lr=lr, weight_decay=weight_decay,
            extra_params=list(self.input_stage.parameters()),
        )

    def summary(self):
        print("GeoLIPEncoder")
        print("=" * 50)
        param_count(self.input_stage, "input")
        self.enc.summary()
        total = sum(p.numel() for p in self.parameters())
        print(f"  Grand total: {total:,}")
        return total