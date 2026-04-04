"""
GeometricTransformer — CM-validated dual-stream with optional flows.

Stack of GeometricTransformerLayers with:
    - CM-gated observation at every layer
    - Optional FlowEnsemble at every layer (config-driven)
    - Cross-layer Cayley rotation on hidden states
    - Built-in geometric regularization via geometric_losses()

CRITICAL: precompute_cm_gates() has @torch.no_grad() + @torch.compiler.disable.
          update_nce_bank() has @torch.no_grad(). See CLAUDE.md rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.core.distinguish.losses import (
    observer_loss as _geolip_observer_loss,
    ce_loss_paired as _geolip_ce_loss_paired,
    spread_loss as _geolip_spread_loss,
)
from geolip_core.pipeline.observer import BaseTower

from .curate_cm_validated import anchor_neighborhood_cm
from .layer_geometric import GeometricTransformerLayer
from .align_cayley import CayleyOrthogonal
from .distinguish_nce_bank import GeoResidualBank

# Optional: geofractal WideRouter for compilation
try:
    from geofractal.router.wide_router import WideRouter
    _HAS_WIDE_ROUTER = True
except ImportError:
    _HAS_WIDE_ROUTER = False


class GeometricTransformer(BaseTower):
    """Geometric Transformer — CM-validated dual-stream with optional flows.

    Stack of GeometricTransformerLayers with:
        - CM-gated observation at every layer
        - Optional FlowEnsemble at every layer (config-driven)
        - Cross-layer Cayley rotation on hidden states
        - Built-in geometric regularization via geometric_losses()
    """
    def __init__(self, name, d_model=512, n_heads=8, n_layers=4,
                 n_anchors=32, manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cross_layer_rotation=True, cm_neighbors=3,
                 nce_bank_size=4096, nce_temperature=0.1,
                 vocab_size=None, max_seq_len=2048,
                 flow_keys=None, flow_fusion='weighted'):
        super().__init__(name)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_anchors = n_anchors
        self._pw_dim = n_comp * d_comp

        if vocab_size is not None:
            self.attach('embed', nn.Embedding(vocab_size, d_model))
            self.attach('pos_embed', nn.Embedding(max_seq_len, d_model))
            self.attach('head', nn.Linear(d_model, vocab_size, bias=False))

        for i in range(n_layers):
            self.attach(f'layer_{i}', GeometricTransformerLayer(
                f'{name}_L{i}', d_model, n_heads, n_anchors,
                manifold_dim, n_comp, d_comp, context_dim, quat_dim,
                dropout, cm_neighbors,
                flow_keys=flow_keys, flow_fusion=flow_fusion))

        if cross_layer_rotation and n_layers > 1:
            for i in range(n_layers - 1):
                self.attach(f'cross_rot_{i}', CayleyOrthogonal(
                    f'{name}_xrot_{i}', d_model))

        self.attach('final_norm', nn.LayerNorm(d_model))

        # Cross-stream contrastive (CLIP-style)
        if nce_bank_size > 0:
            nce_proj_dim = 128
            self.attach('nce_content_proj', nn.Sequential(
                nn.Linear(d_model, nce_proj_dim),
                nn.GELU(),
                nn.Linear(nce_proj_dim, nce_proj_dim),
            ))
            self.attach('nce_geo_proj', nn.Sequential(
                nn.Linear(self._pw_dim, nce_proj_dim),
                nn.GELU(),
                nn.Linear(nce_proj_dim, nce_proj_dim),
            ))
            self.attach('nce_bank', GeoResidualBank(
                nce_proj_dim, bank_size=nce_bank_size,
                temperature=nce_temperature))

        self._config = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            n_anchors=n_anchors, manifold_dim=manifold_dim,
            n_comp=n_comp, d_comp=d_comp, context_dim=context_dim,
            quat_dim=quat_dim, dropout=dropout,
            cross_layer_rotation=cross_layer_rotation,
            cm_neighbors=cm_neighbors, vocab_size=vocab_size,
            nce_bank_size=nce_bank_size, nce_temperature=nce_temperature,
            flow_keys=flow_keys, flow_fusion=flow_fusion,
        )

    @property
    def config(self):
        return self._config.copy()

    def invalidate_caches(self):
        """Invalidate all cuSOLVER-dependent caches. Call after optimizer.step()."""
        for i in range(self.n_layers):
            self[f'layer_{i}']['cm_gate'].invalidate_cache()
            self[f'layer_{i}']['rotation'].invalidate_cache()
        # Cross-layer rotations
        for name in list(self.components.keys()):
            if name.startswith('cross_rot'):
                self[name].invalidate_cache()

    @torch.no_grad()
    @torch.compiler.disable
    def precompute_cm_gates(self):
        """Precompute ALL cuSOLVER-dependent operations for all layers.

        Caches: CM gate anchor quality (det) + Cayley rotations (solve).
        Must be called BEFORE the compiled forward pass. CUDA graph
        capture cannot contain cuSOLVER calls.

        Idempotent: skips components with warm caches.
        """
        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            # CM gate — uses det
            anchors_n = F.normalize(
                layer['observer'].association.constellation.anchors, dim=-1)
            layer['cm_gate'].precompute(anchors_n.detach())
            # Per-layer Cayley rotation — uses solve
            layer['rotation'].precompute()
        # Cross-layer rotations — uses solve
        for name in list(self.components.keys()):
            if name.startswith('cross_rot'):
                self[name].precompute()

    def geometric_losses(self, cv_target=0.215, cv_weight=0.1, spread_weight=0.01):
        """Compute geometric regularization from current anchor geometry."""
        total_cv = torch.tensor(0.0)
        total_spread = torch.tensor(0.0)
        n = 0

        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            anchors = layer['observer'].association.constellation.anchors
            anchors_n = F.normalize(anchors, dim=-1)
            A = anchors_n.shape[0]

            if n == 0:
                total_cv = total_cv.to(anchors.device)
                total_spread = total_spread.to(anchors.device)

            cos = anchors_n @ anchors_n.T
            idx = torch.triu_indices(A, A, offset=1, device=cos.device)
            pairwise_dist = 1.0 - cos[idx[0], idx[1]]
            cv = pairwise_dist.std() / (pairwise_dist.mean() + 1e-8)
            total_cv = total_cv + (cv - cv_target).pow(2)

            mask = ~torch.eye(A, dtype=torch.bool, device=cos.device)
            total_spread = total_spread + F.relu(cos[mask]).mean()

            n += 1

        losses = {}
        if n > 0:
            losses['cv'] = cv_weight * total_cv / n
            losses['spread'] = spread_weight * total_spread / n
            losses['geo_total'] = losses['cv'] + losses['spread']
        return losses

    def infonce_loss(self, cls_index=0):
        """Cross-stream contrastive: content queries against decoupled geometry."""
        if not self.has('nce_bank'):
            return {}

        hidden = getattr(self, '_last_hidden', None)
        geo_residual = getattr(self, '_last_geo_residual', None)
        if hidden is None or geo_residual is None:
            return {}

        content_cls = self['nce_content_proj'](hidden[:, cls_index])
        geo_cls = self['nce_geo_proj'](geo_residual[:, cls_index].detach())

        loss, acc = self['nce_bank'](content_cls, geo_cls)
        return {'nce': loss, 'nce_acc': acc}

    @torch.no_grad()
    def update_nce_bank(self, cls_index=0):
        """Enqueue projected geo keys into bank. Call AFTER backward."""
        if not self.has('nce_bank') or not self.has('nce_geo_proj'):
            return

        geo_residual = getattr(self, '_last_geo_residual', None)
        if geo_residual is None:
            return

        geo_cls = self['nce_geo_proj'](geo_residual[:, cls_index].detach())
        self['nce_bank'].enqueue(F.normalize(geo_cls, dim=-1))

    def anchor_diagnostics(self):
        """Per-layer anchor health diagnostics."""
        diag = {}
        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            anchors = layer['observer'].association.constellation.anchors
            anchors_n = F.normalize(anchors.detach(), dim=-1)
            A = anchors_n.shape[0]

            cos = anchors_n @ anchors_n.T
            idx = torch.triu_indices(A, A, offset=1, device=cos.device)
            pairwise = 1.0 - cos[idx[0], idx[1]]
            cv = (pairwise.std() / (pairwise.mean() + 1e-8)).item()

            with torch.no_grad():
                anchor_cm, _ = anchor_neighborhood_cm(
                    anchors_n, layer['cm_gate'].n_neighbors)

            diag[f'layer_{i}'] = {
                'anchor_cv': cv,
                'mean_pairwise_dist': pairwise.mean().item(),
                'min_pairwise_dist': pairwise.min().item(),
                'cm_positive_frac': (anchor_cm > 0).float().mean().item(),
                'cm_mean': anchor_cm.mean().item(),
                'cm_std': anchor_cm.std().item(),
            }
        return diag

    def param_report(self):
        total = 0
        name = getattr(self, '_tower_name', self.__class__.__name__)
        print(f"\n  {name} — parameter report (CM-validated + flows)")
        print(f"  {'Component':<35s}  {'Params':>12s}")
        print(f"  {'-'*35}  {'-'*12}")
        for cname, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            total += n
            print(f"  {cname:<35s}  {n:>12,}")
        print(f"  {'-'*35}  {'-'*12}")
        print(f"  {'TOTAL':<35s}  {total:>12,}")
        return total


    def forward(self, x, attn_mask=None, key_padding_mask=None,
                return_geo_state=False):
        if self.has('embed') and x.dtype in (torch.long, torch.int32, torch.int64):
            pos = torch.arange(x.shape[1], device=x.device)
            x = self['embed'](x) + self['pos_embed'](pos)

        geo_states = []
        has_xrot = self.has('cross_rot_0')
        geo_residual = None

        for i in range(self.n_layers):
            x, geo_residual, geo_state = self[f'layer_{i}'](
                x, geo_residual=geo_residual,
                attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            if return_geo_state:
                geo_states.append(geo_state)
            if has_xrot and i < self.n_layers - 1:
                x = self[f'cross_rot_{i}'](x)

        # Stash for NCE bank if active (outside compiled graph only)
        if self.has('nce_bank'):
            self._last_geo_residual = geo_residual
            self._last_hidden = x

        x = self['final_norm'](x)
        if self.has('head'):
            x = self['head'](x)

        return (x, geo_states) if return_geo_state else x

    # ── Paired forward + observer loss ──────────────────────────────

    def _run_view(self, x, attn_mask=None, key_padding_mask=None):
        """Run one view through the full pipeline.
        Retains ALL layers' geo_states — every layer needs gradient.
        """
        has_xrot = self.has('cross_rot_0')
        geo_residual = None

        if self.has('embed') and x.dtype in (torch.long, torch.int32, torch.int64):
            pos = torch.arange(x.shape[1], device=x.device)
            x = self['embed'](x) + self['pos_embed'](pos)

        geo_states = []
        for i in range(self.n_layers):
            x, geo_residual, geo_state = self[f'layer_{i}'](
                x, geo_residual=geo_residual,
                attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            geo_states.append(geo_state)
            if has_xrot and i < self.n_layers - 1:
                x = self[f'cross_rot_{i}'](x)

        x = self['final_norm'](x)
        return x, geo_states

    def forward_paired(self, x1, x2, cls_index=0,
                       attn_mask=None, key_padding_mask=None):
        """Dual-view forward for observer loss training.

        Observer loss reads FINAL layer's observations (coherent space).
        Non-final layers get gradient through the geo_residual stream
        (FiLM → context → history_mlp → geo_residual → earlier layers).
        All layers' computation graphs are retained by _run_view.
        """
        B = x1.shape[0]
        x_cat = torch.cat([x1, x2], dim=0)
        feat_cat, geo_states = self._run_view(x_cat, attn_mask, key_padding_mask)

        c = cls_index
        gs = geo_states[-1]  # final layer — coherent representation space

        return {
            'embedding':      gs['embedding'][:B, c],
            'embedding_aug':  gs['embedding'][B:, c],
            'patchwork1':     gs['patchwork'][:B, c],
            'patchwork1_aug': gs['patchwork'][B:, c],
            'bridge1':        gs['bridge'][:B, c],
            'bridge2':        gs['bridge'][B:, c],
            'assign1':        gs['assignment'][:B, c],
            'assign2':        gs['assignment'][B:, c],
            'cos1':           gs['cos_to_anchors'][:B, c],
            'tri1':           gs['triangulation'][:B, c],
            'tri2':           gs['triangulation'][B:, c],
            'features1':      feat_cat[:B],
            'features2':      feat_cat[B:],
            'gate_values1':   gs['gate_values'][:B, c],
            'gate_values2':   gs['gate_values'][B:, c],
            'cm_quality1':    gs['cm_quality'][:B],
            'cm_quality2':    gs['cm_quality'][B:],
        }

    def compute_loss(self, output, targets, cls_index=0,
                     w_ce=1.0, head=None, **loss_kwargs):
        final_layer = self[f'layer_{self.n_layers - 1}']
        anchors = final_layer['observer'].association.constellation.anchors

        obs_loss, ld = _geolip_observer_loss(
            output, anchors=anchors, targets=targets,
            **loss_kwargs)

        if head is not None:
            feat1 = output['features1'][:, cls_index]
            feat2 = output['features2'][:, cls_index]
            logits1 = head(feat1)
            logits2 = head(feat2)
            l_ce, acc = _geolip_ce_loss_paired(logits1, logits2, targets)
            ld['ce'], ld['acc'] = l_ce, acc
            ld['logits'] = logits1
            loss = w_ce * l_ce + obs_loss
            ld['loss_task'] = l_ce.detach()
        else:
            loss = obs_loss

        ld['loss_observer'] = obs_loss.detach()

        w_spread = loss_kwargs.get('w_spread', 0.01)
        if self.n_layers > 1 and w_spread > 0:
            other_spread = torch.tensor(0.0, device=anchors.device)
            for i in range(self.n_layers - 1):
                layer = self[f'layer_{i}']
                layer_anchors = layer['observer'].association.constellation.anchors
                other_spread = other_spread + _geolip_spread_loss(layer_anchors)
            other_spread = w_spread * other_spread / (self.n_layers - 1)
            loss = loss + other_spread
            ld['spread_other_layers'] = other_spread.detach()

        ld['total'] = loss
        return loss, ld


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def geo_transformer_esm2(name='geo_esm2', n_layers=6, **kw):
    """Pre-configured for ESM-2 650M (d=1280)."""
    return GeometricTransformer(name, d_model=1280, n_heads=16,
        n_layers=n_layers, n_anchors=32, manifold_dim=256,
        n_comp=8, d_comp=32, context_dim=128, quat_dim=64, **kw)

def geo_transformer_small(name='geo_small', n_layers=4, **kw):
    """Small config for prototyping."""
    return GeometricTransformer(name, d_model=256, n_heads=8,
        n_layers=n_layers, n_anchors=16, manifold_dim=128,
        n_comp=4, d_comp=16, context_dim=64, quat_dim=32, **kw)

def geo_transformer_vision(name='geo_vit', n_layers=4, **kw):
    """For scatter/SVD vision pipeline (patches as tokens)."""
    return GeometricTransformer(name, d_model=384, n_heads=8,
        n_layers=n_layers, n_anchors=32, manifold_dim=128,
        n_comp=8, d_comp=16, context_dim=64, quat_dim=32, **kw)
