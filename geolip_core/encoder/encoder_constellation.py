"""
Constellation Encoder — Full-Stack GeoLIP Composition
========================================================
Subclasses GeoLIP. Registers constellation stages. Defines the
canonical observation pipeline using those stages.

The partial encoders (conv, wavelet, scatter, transformer) are Input
stages. They produce embeddings on S^(d-1). This module consumes those
embeddings through the paradigm:

    Association → Mutation → Curation → (recurse or) → Distinction

Every component is a registered stage accessed by name. Swap any stage
by re-registering. Add tiers by registering additional association/curation
pairs. The paradigm holds.

ConstellationEncoder:   GeoLIP with constellation stages pre-registered.
ClassificationHead:     Distinction stage for supervised classification.
GeoLIPEncoder:          Any Input + ConstellationEncoder composed.

Usage:
    from geolip_core.encoder import ConstellationEncoder, GeoLIPEncoder, ConvEncoder

    # Full stack from raw embeddings
    enc = ConstellationEncoder(dim=384, n_anchors=512, num_classes=100)
    out = enc.forward_paired(emb1, emb2, raw_mag1, raw_mag2)
    loss, ld = enc.compute_loss(out, targets)

    # Composed with any Input stage
    model = GeoLIPEncoder(input_stage=ConvEncoder(384), dim=384, num_classes=100)
    out = model.forward_paired(view1, view2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.observer import GeoLIP, Distinction
from ..core.constellation import ConstellationAssociation, ConstellationCuration
from ..core.patchwork import MagnitudeFlow
from ..core.activation import make_activation
from ..core.core import param_count, model_summary
from ..core.losses import ce_loss_paired, observer_loss


# ══════════════════════════════════════════════════════════════════
# CLASSIFICATION HEAD — Distinction stage
# ══════════════════════════════════════════════════════════════════

class ClassificationHead(Distinction):
    """Distinction stage for supervised classification.

    Reads curated features → logits. This is the only stage that
    knows about the downstream task (class count).

    Args:
        feature_dim: input feature dimension (from curation)
        num_classes: output classes
        hidden_dim: intermediate MLP dimension
        activation: activation function name
        dropout: dropout rate
    """

    def __init__(self, feature_dim, num_classes, hidden_dim=None,
                 activation='squared_relu', dropout=0.1):
        super().__init__()
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
# CONSTELLATION ENCODER — GeoLIP with constellation stages
# ══════════════════════════════════════════════════════════════════

class ConstellationEncoder(GeoLIP):
    """GeoLIP pre-configured with constellation observation stages.

    Registers:
        mutation 'magnitude':       MagnitudeFlow (relay-stack confidence)
        association 'constellation': ConstellationAssociation (triangulate)
        curation 'constellation':   ConstellationCuration (patchwork + bridge)
        distinction 'classify':     ClassificationHead (optional, if num_classes > 0)

    The forward implements the canonical pipeline using registered stages.
    Swap any stage by re-registering with add_*. Add observation tiers
    by registering additional association/curation pairs.

    Args:
        dim: embedding dimension on S^(dim-1)
        n_anchors: constellation anchors
        n_comp: patchwork compartments
        d_comp: per-compartment hidden dim
        num_classes: classification targets (0 = no distinction stage)
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
        super().__init__(dim)
        self.cv_target = cv_target
        self.infonce_temp = infonce_temp
        self.assign_temp = assign_temp
        self.config = {k: v for k, v in locals().items()
                       if k != 'self' and not k.startswith('_')}

        # ── Register stages through the paradigm ──

        self.add_mutation('magnitude', MagnitudeFlow(
            dim=dim, n_anchors=n_anchors,
            hidden_dim=mag_hidden, n_heads=mag_heads, n_layers=mag_layers,
            mag_min=mag_min, mag_max=mag_max, n_comp=n_comp,
        ))

        self.add_association('constellation', ConstellationAssociation(
            dim=dim, n_anchors=n_anchors,
            anchor_drop=anchor_drop, assign_temp=assign_temp,
        ))

        self.add_curation('constellation', ConstellationCuration(
            n_anchors=n_anchors, dim=dim,
            n_comp=n_comp, d_comp=d_comp, activation=activation,
        ))

        if num_classes > 0:
            curation = self.curations['constellation']
            self.add_distinction('classify', ClassificationHead(
                feature_dim=curation.feature_dim,
                num_classes=num_classes,
                hidden_dim=curation.patchwork.output_dim,
                activation=activation,
            ))

        # Buffers for push compatibility
        self.register_buffer('anchor_classes',
                             torch.zeros(n_anchors, dtype=torch.long))
        self.register_buffer('class_centroids',
                             torch.zeros(num_classes if num_classes > 0 else 1, dim))

    # ── Stage accessors ──

    @property
    def constellation(self):
        """The anchor primitive."""
        return self.associations['constellation'].constellation

    @property
    def patchwork(self):
        return self.curations['constellation'].patchwork

    @property
    def mag_flow(self):
        return self.mutations['magnitude']

    # ── Magnitude context ──

    def _compute_magnitude(self, emb, raw_magnitude):
        """Use the magnitude mutation to produce per-anchor confidence."""
        mag_flow = self.mutations['magnitude']
        anchors_n = F.normalize(self.constellation.anchors, dim=-1)
        tri = emb @ anchors_n.T
        return mag_flow(emb, tri, raw_magnitude)

    # ── Observation pipeline ──

    def _observe(self, emb, mag=None):
        """Single observation tier: associate → curate."""
        assoc = self.associations['constellation']
        curate = self.curations['constellation']

        a_out = assoc(emb, mag=mag)
        c_out = curate.curate_full(a_out, emb=emb)

        return {
            'embedding': emb,
            'features': c_out['features'],
            'triangulation': a_out['distances'],
            'cos_to_anchors': a_out['cos_to_anchors'],
            'nearest': a_out['nearest'],
            'assignment': a_out['assignment'],
            'patchwork': c_out['patchwork'],
            'bridge': c_out['bridge'],
        }

    def _observe_paired(self, emb1, emb2, mag1=None, mag2=None):
        """Paired observation: two views through the same stages."""
        assoc = self.associations['constellation']
        curate = self.curations['constellation']

        a1 = assoc(emb1, mag=mag1)
        a2 = assoc(emb2, mag=mag2)
        c1 = curate.curate_full(a1, emb=emb1)
        c2 = curate.curate_full(a2, emb=emb2)

        return {
            'embedding': emb1, 'embedding_aug': emb2,
            'mag1': mag1, 'mag2': mag2,
            'features1': c1['features'], 'features2': c2['features'],
            'cos1': a1['cos_to_anchors'], 'cos2': a2['cos_to_anchors'],
            'tri1': a1['distances'], 'tri2': a2['distances'],
            'nearest': a1['nearest'],
            'assign1': a1['assignment'], 'assign2': a2['assignment'],
            'patchwork1': c1['patchwork'], 'patchwork1_aug': c2['patchwork'],
            'bridge1': c1['bridge'], 'bridge2': c2['bridge'],
        }

    # ── Forward: the canonical pipeline ──

    def forward(self, emb, raw_magnitude=None):
        """Single view: emb on S^(d-1) → observation → distinction.

        Args:
            emb: (B, dim) L2-normalized
            raw_magnitude: (B, 1) pre-norm magnitude. None → 1.0.
        """
        if raw_magnitude is None:
            raw_magnitude = torch.ones(emb.shape[0], 1, device=emb.device, dtype=emb.dtype)

        # Mutation: magnitude context
        mag, mag_comp = self._compute_magnitude(emb, raw_magnitude)

        # Association → Curation
        out = self._observe(emb, mag=mag)
        out['mag_comp'] = mag_comp

        # Distinction (if registered)
        if 'classify' in self.distinctions:
            out['logits'] = self.distinctions['classify'](out['features'])

        return out

    def forward_paired(self, emb1, emb2, raw_mag1=None, raw_mag2=None):
        """Two views: paired observation → distinction on both.

        Args:
            emb1, emb2: (B, dim) L2-normalized
            raw_mag1, raw_mag2: (B, 1) pre-norm magnitudes
        """
        ones = lambda e: torch.ones(e.shape[0], 1, device=e.device, dtype=e.dtype)
        if raw_mag1 is None: raw_mag1 = ones(emb1)
        if raw_mag2 is None: raw_mag2 = ones(emb2)

        # Mutation: magnitude context for both views
        mag1, mc1 = self._compute_magnitude(emb1, raw_mag1)
        mag2, mc2 = self._compute_magnitude(emb2, raw_mag2)

        # Paired association → curation
        out = self._observe_paired(emb1, emb2, mag1=mag1, mag2=mag2)
        out['mag_comp1'] = mc1
        out['mag_comp2'] = mc2

        # Distinction on both views
        if 'classify' in self.distinctions:
            classify = self.distinctions['classify']
            out['logits'] = classify(out['features1'])
            out['logits_aug'] = classify(out['features2'])

        return out

    # ── Loss: observer + optional task ──

    def compute_loss(self, output, targets, w_ce=1.0, **loss_kwargs):
        """Observer self-organization + classification.

        Observer loss owns geometric + internal domains.
        CE is applied here only if a distinction stage is registered.

        Returns: (total_loss, loss_dict)
        """
        obs_loss, ld = observer_loss(
            output,
            anchors=self.constellation.anchors,
            targets=targets,
            infonce_temp=self.infonce_temp,
            assign_temp=self.assign_temp,
            cv_target=self.cv_target,
            **loss_kwargs,
        )

        if 'classify' in self.distinctions and 'logits' in output:
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

    # ── Anchor parameter management ──

    def get_anchor_param_ids(self):
        """Param ids that must have weight_decay=0."""
        ids = set(id(p) for p in self.constellation.parameters())
        for relay in self.mag_flow.relays:
            ids.add(id(relay.anchors))
        return ids

    def make_optimizer(self, lr=3e-4, weight_decay=0.05, extra_params=None):
        """AdamW with anchor exclusion from weight decay.

        Args:
            extra_params: additional params (e.g. from an Input stage)
        """
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
        print("ConstellationEncoder Summary")
        print("=" * 50)
        for cat_name, container in [
            ('mutations', self.mutations),
            ('associations', self.associations),
            ('curations', self.curations),
            ('distinctions', self.distinctions),
        ]:
            for name, stage in container.items():
                param_count(stage, f"{cat_name}.{name}")
        print("-" * 50)
        return model_summary(self)


# ══════════════════════════════════════════════════════════════════
# GEOLIP ENCODER — Input + ConstellationEncoder
# ══════════════════════════════════════════════════════════════════

class GeoLIPEncoder(nn.Module):
    """Any Input stage + ConstellationEncoder.

    Composes an Input (pixels/wavelets/tokens → S^(d-1)) with
    a ConstellationEncoder (observation pipeline → output).

    The Input stage is registered on the ConstellationEncoder's
    inputs dict, making it accessible through the paradigm.

    Args:
        input_stage: nn.Module implementing Input interface
            (must have .dim property, forward() → (emb, magnitude))
        **kwargs: forwarded to ConstellationEncoder
    """

    def __init__(self, input_stage, **kwargs):
        super().__init__()
        dim = kwargs.pop('dim', input_stage.dim)
        self.enc = ConstellationEncoder(dim=dim, **kwargs)
        self.enc.add_input('primary', input_stage)
        self._init_input_weights()

    @property
    def input_stage(self):
        return self.enc.inputs['primary']

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
        return self.enc.make_optimizer(lr=lr, weight_decay=weight_decay)

    def summary(self):
        print("GeoLIPEncoder Summary")
        print("=" * 50)
        param_count(self.input_stage, "input_stage")
        self.enc.summary()
        print(f"  Grand total: {sum(p.numel() for p in self.parameters()):,}")