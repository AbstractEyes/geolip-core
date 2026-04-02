"""
geolip_core.core.curate.flows — Multi-opinion geometric flow ensemble.

Each flow is a TorchComponent producing a geometric prediction from
(anchors, queries) using a distinct mathematical formulation.
The FlowEnsemble is a BaseTower managing attached flows via the
standard router attach/detach pattern.

Flows are optional, compartmentalizable, and replaceable at runtime:
    ensemble.detach('velocity')
    ensemble.attach('velocity', FlowVelocity('velocity', d, k))

Integration:
    Baked into GeometricTransformerLayer as an optional 5th context stream
    in PositionGeometricContext. Configured via the CONFIG dict.

    config = {
        ...
        'flow_classes': ['quat_lite', 'velocity', 'orbital'],
        'flow_fusion': 'weighted',
        ...
    }

Architecture (geofractal):
    core/curate/flows.py          THIS FILE — TorchComponents doing the math
    pipeline/components/           wired via BaseTower.attach()
    pipeline/layer.py              GeometricTransformerLayer owns a FlowEnsemble

Author: AbstractPhil + Claude Opus 4.6
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

try:
    import geolip_core.linalg as LA
except ImportError:
    import torch.linalg as LA

from geolip_core.pipeline.observer import TorchComponent, BaseTower


# ═══════════════════════════════════════════════════════════════════
# Base Flow (TorchComponent)
# ═══════════════════════════════════════════════════════════════════

class BaseFlow(TorchComponent):
    """Base class for geometric flows.

    Interface contract:
        Input:  anchors [B, k, d], queries [B, n, d]
        Output: prediction [B, n, d], confidence [B, n, 1]

    Subclasses implement _flow(anchors, queries) → [B, n, d].
    Confidence head is shared across all flows.
    """
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name)
        self.d_model = d_model
        self.n_anchors = n_anchors
        self.confidence = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, anchors: Tensor, queries: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self._flow(anchors, queries)
        conf = torch.sigmoid(self.confidence(pred))
        return pred, conf

    def _flow(self, anchors: Tensor, queries: Tensor) -> Tensor:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
# FlowQuaternion — Full MHA quaternion rotation
# ═══════════════════════════════════════════════════════════════════

class FlowQuaternion(BaseFlow):
    """Full multi-head attention → quaternion rotation of queries."""
    def __init__(self, name: str, d_model: int, n_anchors: int, n_heads: int = 4):
        super().__init__(name, d_model, n_anchors)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.quat_proj = nn.Linear(d_model, 4)

    def _flow(self, anchors, queries):
        B, n, d = queries.shape
        k = anchors.shape[1]
        h, hd = self.n_heads, self.head_dim

        Q = self.q_proj(queries).view(B, n, h, hd).transpose(1, 2)
        K = self.k_proj(anchors).view(B, k, h, hd).transpose(1, 2)
        V = self.v_proj(anchors).view(B, k, h, hd).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hd)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).reshape(B, n, d)

        q = F.normalize(self.quat_proj(ctx), dim=-1)
        rotated = self._quat_rotate(queries, q)
        return self.out_proj(ctx + rotated)

    @staticmethod
    def _quat_rotate(v, q):
        w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
        v3 = v[..., :3]
        xyz = torch.cat([x, y, z], dim=-1)
        t = 2.0 * torch.cross(xyz, v3, dim=-1)
        v3_rot = v3 + w * t + torch.cross(xyz, t, dim=-1)
        return torch.cat([v3_rot, v[..., 3:]], dim=-1) if v.shape[-1] > 3 else v3_rot


# ═══════════════════════════════════════════════════════════════════
# FlowQuaternionLite — Centroid-based lightweight quaternion
# ═══════════════════════════════════════════════════════════════════

class FlowQuaternionLite(BaseFlow):
    """Anchor centroid → direct quaternion prediction. No MHA."""
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.anchor_compress = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.quat_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.out_proj = nn.Linear(d_model, d_model)

    def _flow(self, anchors, queries):
        B, n, d = queries.shape
        anchor_ctx = self.anchor_compress(anchors.mean(dim=1, keepdim=True)).expand(B, n, d)
        combined = torch.cat([self.query_proj(queries), anchor_ctx], dim=-1)
        q = F.normalize(self.quat_head(combined), dim=-1)
        return self.out_proj(FlowQuaternion._quat_rotate(queries, q))


# ═══════════════════════════════════════════════════════════════════
# FlowVelocity — Tangent-space angular velocity
# ═══════════════════════════════════════════════════════════════════

class FlowVelocity(BaseFlow):
    """Angular velocity on the tangent bundle. Euler integration."""
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.anchor_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.vel_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.dt = nn.Parameter(torch.tensor(0.1))

    def _flow(self, anchors, queries):
        B, n, d = queries.shape
        a_proj = self.anchor_proj(anchors)
        q_proj = self.query_proj(queries)
        sim = torch.bmm(q_proj, a_proj.transpose(-2, -1))
        weights = F.softmax(sim / math.sqrt(d), dim=-1)
        direction = torch.bmm(weights, a_proj)
        velocity = self.vel_head(direction - q_proj)
        q_norm = F.normalize(queries, dim=-1)
        radial = (velocity * q_norm).sum(dim=-1, keepdim=True) * q_norm
        tangent_vel = velocity - radial
        return queries + self.dt * tangent_vel


# ═══════════════════════════════════════════════════════════════════
# FlowMagnitude — Gram eigenvalue spectrum modulation
# ═══════════════════════════════════════════════════════════════════

class FlowMagnitude(BaseFlow):
    """Gram eigenvalue magnitude spectrum → gated query modulation."""
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.geom_dim = min(n_anchors, 12)
        self.anchor_proj = nn.Linear(d_model, self.geom_dim)
        self.spec_proj = nn.Sequential(
            nn.Linear(self.geom_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.query_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def _flow(self, anchors, queries):
        B, n, d = queries.shape
        a_geom = self.anchor_proj(anchors)
        G = torch.bmm(a_geom.transpose(-2, -1), a_geom)
        eigenvalues, _ = LA.eigh(G, method='torch')
        magnitudes = eigenvalues.abs().sqrt()
        spec_embed = self.spec_proj(magnitudes).unsqueeze(1).expand(B, n, d)
        q_proj = self.query_proj(queries)
        g = torch.sigmoid(self.gate(torch.cat([q_proj, spec_embed], dim=-1)))
        return queries + g * spec_embed


# ═══════════════════════════════════════════════════════════════════
# FlowOrbital — Omega resonance via eigendecomposition
# ═══════════════════════════════════════════════════════════════════

class FlowOrbital(BaseFlow):
    """Orbital resonance: project into eigenbasis, modulate by CV band, project back."""
    def __init__(self, name: str, d_model: int, n_anchors: int,
                 cv_lo: float = 0.20, cv_hi: float = 0.23):
        super().__init__(name, d_model, n_anchors)
        self.geom_dim = min(n_anchors, 12)
        self.cv_lo, self.cv_hi = cv_lo, cv_hi
        self.anchor_proj = nn.Linear(d_model, self.geom_dim)
        self.mode_response = nn.Parameter(torch.ones(self.geom_dim))
        self.query_to_geom = nn.Linear(d_model, self.geom_dim)
        self.geom_to_query = nn.Linear(self.geom_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _flow(self, anchors, queries):
        B, n, d = queries.shape
        a_geom = self.anchor_proj(anchors)
        G = torch.bmm(a_geom.transpose(-2, -1), a_geom)
        eigenvalues, eigenvectors = LA.eigh(G, method='torch')

        in_band = ((eigenvalues >= self.cv_lo) & (eigenvalues <= self.cv_hi)).float()
        near_binding = torch.exp(-10.0 * (eigenvalues - 0.29154).pow(2))
        mode_weight = self.mode_response.unsqueeze(0) * (1.0 + in_band + near_binding)

        q_geom = self.query_to_geom(queries)
        q_eigen = torch.bmm(q_geom, eigenvectors)
        q_modulated = q_eigen * mode_weight.unsqueeze(1)
        q_out = torch.bmm(q_modulated, eigenvectors.transpose(-2, -1))
        return self.out_proj(self.geom_to_query(q_out) + queries)


# ═══════════════════════════════════════════════════════════════════
# FlowAlignment — SVD Procrustes in projected space
# ═══════════════════════════════════════════════════════════════════

class FlowAlignment(BaseFlow):
    """SVD Procrustes rotation in geom_dim projected space."""
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.geom_dim = min(n_anchors, 12)
        self.anchor_proj = nn.Linear(d_model, self.geom_dim)
        self.query_proj = nn.Linear(d_model, self.geom_dim)
        self.geom_to_query = nn.Linear(self.geom_dim, d_model)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def _flow(self, anchors, queries):
        B, n, d = queries.shape
        a_proj = self.anchor_proj(anchors)
        q_proj = self.query_proj(queries)
        sim = torch.bmm(q_proj, a_proj.transpose(-2, -1)) / math.sqrt(self.geom_dim)
        weights = F.softmax(sim, dim=-1)
        targets = torch.bmm(weights, a_proj)
        C = torch.bmm(q_proj.transpose(-2, -1), targets)
        U, _, Vh = LA.svd(C, method='gram_eigh')
        R = torch.bmm(U, Vh)
        q_rotated = torch.bmm(q_proj, R)
        delta = self.geom_to_query(q_rotated - q_proj)
        return queries + self.strength * delta


# ═══════════════════════════════════════════════════════════════════
# Flow Registry — string-key lookup for config-driven construction
# ═══════════════════════════════════════════════════════════════════

FLOW_REGISTRY: Dict[str, type] = {
    'quaternion': FlowQuaternion,
    'quat_lite': FlowQuaternionLite,
    'velocity': FlowVelocity,
    'magnitude': FlowMagnitude,
    'orbital': FlowOrbital,
    'alignment': FlowAlignment,
}


def build_flow(key: str, name: str, d_model: int, n_anchors: int, **kwargs) -> BaseFlow:
    """Construct a flow from registry key.

    Args:
        key: registry key ('quaternion', 'orbital', etc.)
        name: TorchComponent name for router
        d_model: embedding dimension
        n_anchors: constellation anchor count
        **kwargs: forwarded to flow constructor
    """
    if key not in FLOW_REGISTRY:
        raise ValueError(f"Unknown flow '{key}'. Available: {list(FLOW_REGISTRY.keys())}")
    return FLOW_REGISTRY[key](name, d_model, n_anchors, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# FlowEnsemble (BaseTower) — manages flows via attach/detach
# ═══════════════════════════════════════════════════════════════════

class FlowEnsemble(BaseTower):
    """Multi-flow ensemble as a BaseTower.

    Each flow is a named TorchComponent, attachable and detachable
    at runtime via the standard router pattern:

        ensemble.attach('orbital', FlowOrbital('orbital', 128, 32))
        ensemble.detach('orbital')

    Fusion modes:
        weighted: confidence-softmax weighted average
        gated:    concatenate → learned projection
        residual: confidence-weighted residual sum

    Config-driven construction:
        ensemble = FlowEnsemble.from_config('flows', config)

    Args:
        name: BaseTower name for router
        d_model: embedding dimension
        flow_keys: list of registry keys to instantiate
        n_anchors: constellation anchor count
        fusion: 'weighted' | 'gated' | 'residual'
    """
    def __init__(self, name: str, d_model: int, n_anchors: int,
                 flow_keys: List[str] = None, fusion: str = 'weighted'):
        super().__init__(name)
        self.d_model = d_model
        self.n_anchors = n_anchors
        self.fusion_mode = fusion
        self._flow_names: List[str] = []

        if flow_keys is None:
            flow_keys = []

        for key in flow_keys:
            flow_name = f'flow_{key}'
            flow = build_flow(key, flow_name, d_model, n_anchors)
            self.attach(flow_name, flow)
            self._flow_names.append(flow_name)

        # Per-flow learnable temperature
        self.temperature = nn.Parameter(torch.ones(max(len(flow_keys), 1)))

        # Gated fusion needs a projection
        if fusion == 'gated' and len(flow_keys) > 0:
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model * len(flow_keys), d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    @classmethod
    def from_config(cls, name: str, config: dict) -> 'FlowEnsemble':
        """Build from training config dict.

        Expected config keys:
            flow_classes: list of str ('quat_lite', 'velocity', 'orbital', ...)
            flow_fusion: str ('weighted', 'gated', 'residual')
            manifold_dim: int (used as d_model for flows)
            n_anchors: int
        """
        flow_keys = config.get('flow_classes', [])
        fusion = config.get('flow_fusion', 'weighted')
        d_model = config.get('manifold_dim', 128)
        n_anchors = config.get('n_anchors', 32)
        return cls(name, d_model, n_anchors, flow_keys=flow_keys, fusion=fusion)

    @property
    def n_flows(self) -> int:
        return len(self._flow_names)

    @property
    def active_flow_names(self) -> List[str]:
        return [n for n in self._flow_names if self.has(n)]

    def attach_flow(self, key: str, **kwargs):
        """Attach a new flow by registry key. Inherits device from ensemble."""
        flow_name = f'flow_{key}'
        flow = build_flow(key, flow_name, self.d_model, self.n_anchors, **kwargs)
        # Inherit device from existing parameters
        try:
            device = next(self.parameters()).device
            flow = flow.to(device)
        except StopIteration:
            pass  # no existing params — leave on default device
        self.attach(flow_name, flow)
        if flow_name not in self._flow_names:
            self._flow_names.append(flow_name)
        # Resize temperature
        n = len(self.active_flow_names)
        if n > self.temperature.shape[0]:
            old = self.temperature.data
            self.temperature = nn.Parameter(torch.ones(n, device=old.device))
            self.temperature.data[:old.shape[0]] = old

    def detach_flow(self, key: str):
        """Detach a flow by registry key."""
        flow_name = f'flow_{key}'
        if self.has(flow_name):
            self.detach(flow_name)

    def forward(self, anchors: Tensor, queries: Tensor) -> Tensor:
        """Run all active flows, fuse predictions.

        Args:
            anchors: [B, k, d] constellation anchors
            queries: [B, n, d] query embeddings on S^(d-1)

        Returns:
            fused: [B, n, d] ensemble prediction
        """
        active = self.active_flow_names
        if not active:
            return queries  # no flows attached — identity

        predictions = []
        confidences = []

        for idx, flow_name in enumerate(active):
            pred, conf = self[flow_name](anchors, queries)
            predictions.append(pred)
            temp_idx = min(idx, self.temperature.shape[0] - 1)
            confidences.append(conf * self.temperature[temp_idx])

        if self.fusion_mode == 'weighted':
            return self._weighted_fusion(predictions, confidences)
        elif self.fusion_mode == 'gated':
            return self._gated_fusion(predictions)
        elif self.fusion_mode == 'residual':
            return self._residual_fusion(predictions, confidences, queries)
        else:
            return self._weighted_fusion(predictions, confidences)

    def _weighted_fusion(self, preds, confs):
        conf_stack = torch.cat(confs, dim=-1)
        weights = F.softmax(conf_stack, dim=-1)
        pred_stack = torch.stack(preds, dim=-1)
        return (pred_stack * weights.unsqueeze(-2)).sum(dim=-1)

    def _gated_fusion(self, preds):
        cat = torch.cat(preds, dim=-1)
        return self.gate_proj(cat)

    def _residual_fusion(self, preds, confs, queries):
        conf_stack = torch.cat(confs, dim=-1)
        weights = F.softmax(conf_stack, dim=-1)
        residuals = torch.stack([p - queries for p in preds], dim=-1)
        return queries + (residuals * weights.unsqueeze(-2)).sum(dim=-1)

    def diagnostics(self, anchors: Tensor, queries: Tensor) -> dict:
        """Per-flow diagnostics. Pure tensor ops — compile safe."""
        diag = {}
        for flow_name in self.active_flow_names:
            pred, conf = self[flow_name](anchors, queries)
            diag[flow_name] = {
                'pred_norm': pred.norm(dim=-1).mean(),
                'confidence_mean': conf.mean(),
                'confidence_std': conf.std(),
                'residual_norm': (pred - queries).norm(dim=-1).mean(),
            }
        return diag