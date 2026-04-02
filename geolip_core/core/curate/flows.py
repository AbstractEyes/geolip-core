"""
geolip_core.core.curate.flows — Anchor-space geometric flow opinions.

Each flow produces a [N, A] opinion tensor: its mathematical perspective
on query-anchor relevance. Same shape as triangulation. Same interface.
The pipeline already knows how to consume it.

Flow opinions feed into PositionGeometricContext as the 5th stream,
alongside triangulation, assignment, patchwork, gate_values, and
geometric residual. The context fuse learns to weight flow opinions
against the other geometric signals.

Each flow answers: "from MY mathematical perspective, how relevant
is anchor j for query i?"

  QuaternionFlow:   rotate query, re-triangulate → rotated relevance
  VelocityFlow:     move query along tangent velocity → predicted relevance
  MagnitudeFlow:    Gram eigenvalue spectrum → spectral anchor weights
  OrbitalFlow:      eigenbasis CV-band resonance → resonance-weighted relevance
  AlignmentFlow:    Procrustes-aligned frame → aligned relevance

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
    """Base class for anchor-space geometric flows.

    Interface:
        Input:  anchors [A, d], queries [N, d], tri [N, A]
        Output: opinion [N, A]

    Each flow produces a [N, A] tensor representing its view of
    query-anchor relevance. Subclasses implement _opinion().
    """
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name)
        self.d_model = d_model
        self.n_anchors = n_anchors

    def forward(self, anchors: Tensor, queries: Tensor, tri: Tensor) -> Tensor:
        """
        Args:
            anchors: [A, d] constellation anchors on S^(d-1)
            queries: [N, d] query embeddings on S^(d-1)
            tri:     [N, A] triangulation distances (1 - cos)

        Returns:
            opinion: [N, A] this flow's view of query-anchor relevance
        """
        return self._opinion(anchors, queries, tri)

    def _opinion(self, anchors: Tensor, queries: Tensor, tri: Tensor) -> Tensor:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
# FlowQuaternion — Rotate query, re-triangulate
# ═══════════════════════════════════════════════════════════════════

class FlowQuaternion(BaseFlow):
    """Predict quaternion rotation per query, measure new anchor distances.

    The rotated query has different triangulation to the anchors.
    This opinion says: "if the query WERE rotated to better fit the
    constellation geometry, here's what the triangulation would look like."
    """
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.quat_head = nn.Sequential(
            nn.Linear(d_model + n_anchors, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def _opinion(self, anchors, queries, tri):
        N, d = queries.shape
        A = anchors.shape[0]
        # Predict rotation from query + current triangulation
        q = F.normalize(self.quat_head(torch.cat([queries, tri], dim=-1)), dim=-1)
        # Apply rotation to first 3 dims
        w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        v3 = queries[:, :3]
        xyz = torch.cat([x, y, z], dim=-1)
        t = 2.0 * torch.cross(xyz, v3, dim=-1)
        v3_rot = v3 + w * t + torch.cross(xyz, t, dim=-1)
        q_rot = queries.clone()
        q_rot[:, :3] = v3_rot
        q_rot = F.normalize(q_rot, dim=-1)
        # Re-triangulate: distance from rotated query to each anchor
        return 1.0 - q_rot @ anchors.T  # [N, A]


# ═══════════════════════════════════════════════════════════════════
# FlowQuaternionLite — Lightweight centroid-based rotation
# ═══════════════════════════════════════════════════════════════════

class FlowQuaternionLite(BaseFlow):
    """Centroid-directed quaternion rotation. No attention, no MHA."""
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.quat_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

    def _opinion(self, anchors, queries, tri):
        # Direction toward anchor centroid
        centroid = F.normalize(anchors.mean(dim=0, keepdim=True), dim=-1)
        diff = centroid.expand_as(queries) - queries
        q = F.normalize(self.quat_head(diff), dim=-1)
        w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        v3 = queries[:, :3]
        xyz = torch.cat([x, y, z], dim=-1)
        t = 2.0 * torch.cross(xyz, v3, dim=-1)
        v3_rot = v3 + w * t + torch.cross(xyz, t, dim=-1)
        q_rot = queries.clone()
        q_rot[:, :3] = v3_rot
        q_rot = F.normalize(q_rot, dim=-1)
        return 1.0 - q_rot @ anchors.T


# ═══════════════════════════════════════════════════════════════════
# FlowVelocity — Tangent velocity → predicted future triangulation
# ═══════════════════════════════════════════════════════════════════

class FlowVelocity(BaseFlow):
    """Move query along learned tangent velocity, re-triangulate.

    "If the query moves in its natural direction on the manifold,
    here's what the anchor distances will become."
    """
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.vel_head = nn.Sequential(
            nn.Linear(d_model + n_anchors, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dt = nn.Parameter(torch.tensor(0.1))

    def _opinion(self, anchors, queries, tri):
        velocity = self.vel_head(torch.cat([queries, tri], dim=-1))
        # Project to tangent space (remove radial component)
        q_norm = F.normalize(queries, dim=-1)
        radial = (velocity * q_norm).sum(dim=-1, keepdim=True) * q_norm
        tangent_vel = velocity - radial
        # Euler step + renormalize to S^(d-1)
        q_moved = F.normalize(queries + self.dt * tangent_vel, dim=-1)
        return 1.0 - q_moved @ anchors.T


# ═══════════════════════════════════════════════════════════════════
# FlowMagnitude — Gram eigenvalue spectrum → anchor weights
# ═══════════════════════════════════════════════════════════════════

class FlowMagnitude(BaseFlow):
    """Gram eigenspectrum tells which geometric modes carry energy.
    Projects this into per-anchor weights that modulate triangulation.

    "From the spectral perspective, how much energy does each anchor
    contribute to the constellation's geometry?"
    """
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.geom_dim = min(n_anchors, 12)
        self.anchor_proj = nn.Linear(d_model, self.geom_dim)
        self.spec_to_anchor = nn.Sequential(
            nn.Linear(self.geom_dim, n_anchors),
            nn.GELU(),
            nn.Linear(n_anchors, n_anchors),
        )

    def _opinion(self, anchors, queries, tri):
        A = anchors.shape[0]
        a_geom = self.anchor_proj(anchors)  # [A, geom_dim]
        G = a_geom.T @ a_geom  # [gd, gd]
        G = G.unsqueeze(0)  # [1, gd, gd]
        eigenvalues, _ = LA.eigh(G, method='fl')  # [1, gd]
        magnitudes = eigenvalues.abs().sqrt().squeeze(0)  # [gd]
        # Spectral profile → per-anchor weight
        anchor_weights = torch.sigmoid(self.spec_to_anchor(magnitudes))  # [A]
        # Modulate triangulation
        return tri * anchor_weights.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
# FlowOrbital — CV-band resonance → mode-aware anchor relevance
# ═══════════════════════════════════════════════════════════════════

class FlowOrbital(BaseFlow):
    """Eigendecomposition of anchor Gram → CV-band-aware relevance.

    Projects queries into eigenbasis, checks which modes are in the
    CV band [0.20, 0.23], derives per-anchor resonance weight.

    "From the ω resonance perspective, which anchors participate in
    the geometrically valid modes?"
    """
    def __init__(self, name: str, d_model: int, n_anchors: int,
                 cv_lo: float = 0.20, cv_hi: float = 0.23):
        super().__init__(name, d_model, n_anchors)
        self.geom_dim = min(n_anchors, 12)
        self.cv_lo, self.cv_hi = cv_lo, cv_hi
        self.anchor_proj = nn.Linear(d_model, self.geom_dim)
        self.mode_to_anchor = nn.Linear(self.geom_dim, n_anchors)

    def _opinion(self, anchors, queries, tri):
        A = anchors.shape[0]
        a_geom = self.anchor_proj(anchors)  # [A, gd]
        G = (a_geom.T @ a_geom).unsqueeze(0)  # [1, gd, gd]
        eigenvalues, eigenvectors = LA.eigh(G, method='fl')  # [1, gd], [1, gd, gd]
        eigenvalues = eigenvalues.squeeze(0)  # [gd]
        eigenvectors = eigenvectors.squeeze(0)  # [gd, gd]

        # CV band resonance: modes in [0.20, 0.23] are geometrically valid
        in_band = ((eigenvalues >= self.cv_lo) & (eigenvalues <= self.cv_hi)).float()
        near_binding = torch.exp(-10.0 * (eigenvalues - 0.29154).pow(2))
        mode_weight = 1.0 + in_band + near_binding  # [gd]

        # Per-anchor resonance: how much each anchor participates in valid modes
        # a_geom @ V gives anchor projections into eigenbasis
        anchor_in_basis = a_geom @ eigenvectors  # [A, gd]
        # Weight by mode validity, sum across modes
        resonance = (anchor_in_basis.abs() * mode_weight.unsqueeze(0)).sum(dim=-1)  # [A]
        anchor_resonance = torch.sigmoid(self.mode_to_anchor(mode_weight))  # [A]

        return tri * anchor_resonance.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
# FlowAlignment — Procrustes-aligned triangulation
# ═══════════════════════════════════════════════════════════════════

class FlowAlignment(BaseFlow):
    """SVD alignment: find optimal rotation from queries to anchors,
    measure triangulation in the aligned frame.

    "If the query frame were optimally rotated to match the anchor
    frame, here's what the triangulation would look like."
    """
    def __init__(self, name: str, d_model: int, n_anchors: int):
        super().__init__(name, d_model, n_anchors)
        self.geom_dim = min(n_anchors, 12)
        self.q_proj = nn.Linear(d_model, self.geom_dim)
        self.a_proj = nn.Linear(d_model, self.geom_dim)
        self.strength = nn.Parameter(torch.tensor(0.5))

    def _opinion(self, anchors, queries, tri):
        N = queries.shape[0]
        A = anchors.shape[0]
        q_g = self.q_proj(queries)   # [N, gd]
        a_g = self.a_proj(anchors)   # [A, gd]
        # Cross-covariance [gd, gd]
        C = (q_g.T @ q_g) + (a_g.T @ a_g)  # symmetric, stable for eigh path
        C = C.unsqueeze(0)
        # C is symmetric — use eigh directly (FL kernel, CUDA-graph-safe)
        eigenvalues, eigenvectors = LA.eigh(C, method='fl')
        # Procrustes rotation from eigenvectors
        U = eigenvectors
        Vh = eigenvectors.transpose(-2, -1)
        R = (U @ Vh).squeeze(0)  # [gd, gd]
        # Align queries in projected space
        q_aligned = q_g @ R  # [N, gd]
        # Distance in aligned space → anchor relevance
        # Use original anchors in projected space for triangulation
        aligned_tri = 1.0 - F.normalize(q_aligned, dim=-1) @ F.normalize(a_g, dim=-1).T
        # Blend with original triangulation
        return self.strength * aligned_tri + (1 - self.strength) * tri


# ═══════════════════════════════════════════════════════════════════
# Flow Registry
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
    if key not in FLOW_REGISTRY:
        raise ValueError(f"Unknown flow '{key}'. Available: {list(FLOW_REGISTRY.keys())}")
    return FLOW_REGISTRY[key](name, d_model, n_anchors, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# FlowEnsemble (BaseTower)
# ═══════════════════════════════════════════════════════════════════

class FlowEnsemble(BaseTower):
    """Fuses multiple anchor-space flow opinions into one [N, A] tensor.

    Each flow produces [N, A] — its view of query-anchor relevance.
    The ensemble learns per-flow weights to combine them.

    Config-driven:
        FlowEnsemble.from_config('flows', config)

    Runtime swappable:
        ensemble.attach_flow('alignment')
        ensemble.detach_flow('velocity')
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

        # Learnable per-flow weight (softmax → fusion weights)
        n_init = max(len(flow_keys), 1)
        self.flow_weights = nn.Parameter(torch.zeros(n_init))

    @classmethod
    def from_config(cls, name: str, config: dict) -> 'FlowEnsemble':
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
        """Attach a new flow by registry key. Inherits device."""
        flow_name = f'flow_{key}'
        flow = build_flow(key, flow_name, self.d_model, self.n_anchors, **kwargs)
        try:
            device = next(self.parameters()).device
            flow = flow.to(device)
        except StopIteration:
            pass
        self.attach(flow_name, flow)
        if flow_name not in self._flow_names:
            self._flow_names.append(flow_name)
        # Resize weights
        n = len(self.active_flow_names)
        if n > self.flow_weights.shape[0]:
            old = self.flow_weights.data
            self.flow_weights = nn.Parameter(torch.zeros(n, device=old.device))
            self.flow_weights.data[:old.shape[0]] = old

    def detach_flow(self, key: str):
        flow_name = f'flow_{key}'
        if self.has(flow_name):
            self.detach(flow_name)

    def forward(self, anchors: Tensor, queries: Tensor, tri: Tensor) -> Tensor:
        """
        Args:
            anchors: [A, d] constellation anchors
            queries: [N, d] query embeddings
            tri:     [N, A] triangulation distances

        Returns:
            opinion: [N, A] fused flow opinion
        """
        active = self.active_flow_names
        if not active:
            return tri  # identity: return original triangulation

        opinions = []
        for flow_name in active:
            opinion = self[flow_name](anchors, queries, tri)
            opinions.append(opinion)

        if len(opinions) == 1:
            return opinions[0]

        # Weighted fusion: learned per-flow weights
        weights = F.softmax(self.flow_weights[:len(opinions)], dim=0)
        stacked = torch.stack(opinions, dim=0)  # [n_flows, N, A]
        return (stacked * weights[:, None, None]).sum(dim=0)  # [N, A]

    def diagnostics(self, anchors: Tensor, queries: Tensor, tri: Tensor) -> dict:
        diag = {}
        for flow_name in self.active_flow_names:
            opinion = self[flow_name](anchors, queries, tri)
            diag[flow_name] = {
                'opinion_mean': opinion.mean(),
                'opinion_std': opinion.std(),
                'diff_from_tri': (opinion - tri).abs().mean(),
            }
        return diag