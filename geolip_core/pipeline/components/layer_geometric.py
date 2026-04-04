"""
GeometricTransformerLayer — one layer of the CM-validated dual-stream
geometric transformer with constellation routing and optional flows.

Pipeline per layer:
    1. ManifoldProjection:  h → emb on S^(d-1)
    2. Association: emb → raw triangulation, cos, assignment
    3. CMValidatedGate: per-anchor CM validity → gate_values
    4. Gated curation: patchwork reads tri * gate_values
    4.5 FlowEnsemble (optional): multi-opinion geometric predictions
    5. PositionGeometricContext: 5 streams → FiLM context
    6. ContentAttention (Stream A): standard MHA
    7. GeometricAttention (Stream B): FiLM(Q,K | geo_ctx)
    8. CayleyOrthogonal: align B → A
    9. QuaternionCompose: w=A, i=aligned_B, j=A-B, k=A*B
   10. Decode + gated residual
   11. CM-conditioned geometric residual accumulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.core.associate.constellation import ConstellationObserver
from geolip_core.core.curate.flows import FlowEnsemble
from geolip_core.pipeline.observer import BaseTower

from .project_manifold import ManifoldProjection
from .curate_cm_validated import CMValidatedGate
from .context_position_geometric import PositionGeometricContext
from .attend_content import ContentAttention
from .attend_geometric import GeometricAttention
from .align_cayley import CayleyOrthogonal
from .compose_quaternion import QuaternionCompose


class GeometricTransformerLayer(BaseTower):
    """One layer of the geometric transformer (CM validated + flows).

    Flows are optional, config-driven, and individually replaceable:
        layer['flows'].attach_flow('alignment')
        layer['flows'].detach_flow('velocity')
    """
    def __init__(self, name, d_model, n_heads=8, n_anchors=32,
                 manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cm_neighbors=3, flow_keys=None, flow_fusion='weighted'):
        super().__init__(name)
        self.d_model = d_model
        self.n_anchors = n_anchors
        self.manifold_dim = manifold_dim

        # 1. Project to manifold
        self.attach('projection', ManifoldProjection(
            f'{name}_proj', d_model, manifold_dim))

        # 2. Constellation observer (association + curation — called decomposed)
        self.attach('observer', ConstellationObserver(
            dim=manifold_dim, n_anchors=n_anchors,
            n_comp=n_comp, d_comp=d_comp))

        # 3. CM validated gate — between association and curation
        self.attach('cm_gate', CMValidatedGate(
            n_anchors=n_anchors, n_neighbors=cm_neighbors))

        # 3.5 Flow ensemble — optional multi-opinion geometric fusion
        if flow_keys:
            self.attach('flows', FlowEnsemble(
                f'{name}_flows', manifold_dim, n_anchors,
                flow_keys=flow_keys, fusion=flow_fusion))
            # Blend weight: how much flow opinions influence curation
            # Starts small → flows fade in as they learn
            self.flow_alpha = nn.Parameter(torch.tensor(0.01))

        # 4. Fuse observation into FiLM context (5 streams)
        pw_dim = self['observer'].curation.patchwork.output_dim
        self.attach('context', PositionGeometricContext(
            f'{name}_ctx', n_anchors, pw_dim, manifold_dim, context_dim))

        # 5. Stream A: content
        self.attach('content', ContentAttention(
            f'{name}_content', d_model, n_heads, dropout))

        # 6. Stream B: geometric
        self.attach('geometric', GeometricAttention(
            f'{name}_geo', d_model, n_heads, context_dim, dropout))

        # 7. Cayley rotation: align B → A
        self.attach('rotation', CayleyOrthogonal(f'{name}_cayley', d_model))

        # 8. Quaternion composition
        self.attach('compose', QuaternionCompose(
            f'{name}_quat', d_model, quat_dim))

        # 9. Decode + output gate
        self.attach('decode', nn.Sequential(
            nn.Linear(quat_dim * 4, d_model), nn.GELU(), nn.LayerNorm(d_model)))
        self.attach('gate', nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()))

        # 10. Geometric residual projection (no learned gate — CM quality decides)
        self._pw_dim = pw_dim
        self.attach('geo_proj', nn.Sequential(
            nn.Linear(pw_dim, pw_dim), nn.LayerNorm(pw_dim)))

    def forward(self, x, geo_residual=None, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (B, L, D) input hidden states
            geo_residual: (B, L, pw_dim) accumulated geometric context,
                          or None for first layer

        Returns:
            x_out: (B, L, D) transformed hidden states
            geo_residual_out: (B, L, pw_dim) updated geometric residual
            geo_state: dict with full geometric state + CM diagnostics
        """
        B, L, D = x.shape

        # ════ 1. Project to manifold ════
        emb = self['projection'](x)  # (B, L, manifold_dim)
        emb_flat = emb.reshape(B * L, -1)

        # ════ 2. Association — raw triangulation ════
        a_out = self['observer'].association(emb_flat)

        # ════ 3. CM Gate — validate anchor measurements ════
        anchors_n = F.normalize(
            self['observer'].association.constellation.anchors, dim=-1)
        # CM gate forward — precompute() must have been called before entering
        # the compiled graph (by GeometricTransformer.precompute_cm_gates())
        gate_values, gate_info = self['cm_gate'](a_out['distances'])

        # ════ 4. Gated curation — patchwork reads validated triangulation ════
        a_out_gated = dict(a_out)

        # ════ 4.5 Flow ensemble — anchor-space geometric opinions ════
        flow_opinion = None
        if self.has('flows'):
            flow_opinion = self['flows'](anchors_n, emb_flat, a_out['distances'])  # [N, A]
            # Blend flow opinion into triangulation: raw + alpha*(flow - raw)
            # flow_alpha starts at 0.01 → 99% raw, 1% flow opinion
            # Gradient: observer_loss → patchwork → distances_weighted → flow_opinion → flows
            alpha = self.flow_alpha.sigmoid()
            blended_tri = a_out['distances'] + alpha * (flow_opinion - a_out['distances'])
            a_out_gated['distances_weighted'] = blended_tri * gate_values
        else:
            a_out_gated['distances_weighted'] = a_out['distances'] * gate_values

        c_out = self['observer'].curation.curate_full(a_out_gated, emb=emb_flat)

        # Build observation dict for context
        obs = {
            'embedding': emb_flat,
            'triangulation': a_out['distances'],
            'cos_to_anchors': a_out['cos_to_anchors'],
            'assignment': a_out['assignment'],
            'nearest': a_out['nearest'],
            'patchwork': c_out['patchwork'],
            'bridge': c_out['bridge'],
        }

        # ════ 5. Build FiLM context — 5 streams ════
        geo_res_flat = geo_residual.reshape(B * L, -1) if geo_residual is not None else None
        geo_ctx_flat = self['context'](
            obs, gate_values=gate_values, geo_residual=geo_res_flat,
            flow_output=flow_opinion)
        geo_ctx = geo_ctx_flat.reshape(B, L, -1)

        # ════ 6. Stream A: content attention ════
        a_out_stream = self['content'](
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # ════ 7. Stream B: geometric attention ════
        b_out = self['geometric'](
            x, geo_ctx, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # ════ 8. Cayley rotation: align B → A ════
        b_aligned = self['rotation'](b_out)

        # ════ 9. Quaternion composition ════
        composed = self['compose'](
            arm_w=a_out_stream, arm_i=b_aligned,
            arm_j=a_out_stream - b_aligned, arm_k=a_out_stream * b_aligned)

        # ════ 10. Decode + gated residual ════
        decoded = self['decode'](composed)
        g = self['gate'](torch.cat([x, decoded], dim=-1))
        x_out = g * decoded + (1 - g) * x

        # ════ 11. CM-conditioned geometric residual accumulation ════
        pw_validated = c_out['patchwork'].reshape(B, L, -1)
        cm_quality = gate_values.mean(dim=-1).reshape(B, L, 1)
        geo_update = self['geo_proj'](pw_validated)

        if geo_residual is None:
            geo_residual_out = cm_quality * geo_update
        else:
            geo_residual_out = geo_residual + cm_quality * geo_update

        # ════ Build geo_state dict ════
        def _unflatten(t):
            if t is None:
                return None
            if t.dim() == 1:
                return t.reshape(B, L)
            return t.reshape(B, L, *t.shape[1:])

        geo_state = {
            'embedding':      emb,
            'geo_ctx':        geo_ctx,
            'triangulation':  _unflatten(a_out['distances']),
            'cos_to_anchors': _unflatten(a_out['cos_to_anchors']),
            'assignment':     _unflatten(a_out['assignment']),
            'nearest':        _unflatten(a_out['nearest']),
            'patchwork':      _unflatten(c_out['patchwork']),
            'bridge':         _unflatten(c_out['bridge']),
            'gate_values':    _unflatten(gate_values),
            'gate_info':      gate_info,
            'cm_quality':     cm_quality,
            'content':        a_out_stream,
            'geometric':      b_out,
            'composed':       composed,
            'geo_residual':   geo_residual_out,
            'flow_opinion':   _unflatten(flow_opinion) if flow_opinion is not None else None,
        }

        return x_out, geo_residual_out, geo_state
