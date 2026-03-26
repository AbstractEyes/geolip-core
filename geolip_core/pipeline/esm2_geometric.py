"""
ESM-2 Geometric Self-Distillation Pipeline.

Freezes ESM-2, taps hidden states at multiple layers, observes
structural relationships via the geometric pipeline, and self-distills
the frozen model's own token logits from geometric features.

Architecture:
    ESM-2 (frozen, hooks at specified layers)
        ↓  hidden states at n_taps layers
    Per-tap projection (1280 → tap_dim)
        ↓  stack → (B, n_taps, tap_dim)
    ObserveSVDTokens (structural decomposition across layers)
    AssociateConstellation (triangulate on S^(d-1))
    CurateCMGate (CM validity selection)
    CuratePatchwork (interpret curated geometry)
    FuseGeometric (unified protein-level feature)
        ↓
    Self-distillation head → predict ESM-2 token logits
    Variant effect head → predict fitness (fine-tune phase)

Usage:
    from geolip_core.pipeline.esm2_geometric import ESM2GeometricPipeline

    pipe = ESM2GeometricPipeline('esm2_geo', esm_model_name='facebook/esm2_t33_650M_UR50D')
    loss, info = pipe(input_ids, attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent

from geolip_core.pipeline.components import (
    ObserveSVDTokens, AssociateConstellation,
    CurateCMGate, CuratePatchwork, FuseGeometric,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAP SYSTEM — extract hidden states from frozen ESM-2
# ═══════════════════════════════════════════════════════════════════════════════

class ESM2TapSystem(TorchComponent):
    """Hooks into frozen ESM-2 and extracts hidden states at specified layers.

    On forward, runs ESM-2 and captures hidden states via hooks.
    Projects each tap from esm_dim → tap_dim. Stacks into (B, n_taps, tap_dim).

    Also captures the final logits for self-distillation target.

    Writes: cache['tap_stack']       — (B, n_taps, tap_dim)
            cache['esm_logits']      — (B, L, vocab) target for distillation
            cache['esm_hidden_final']— (B, L, esm_dim) last layer hidden
            cache['attention_mask']  — (B, L) mask
            cache['seq_len']         — int, sequence length
    """

    def __init__(self, name, esm_model, tap_layers, tap_dim=256, **kwargs):
        super().__init__(name, **kwargs)

        self.esm = esm_model
        self.tap_layers = sorted(tap_layers)
        self.n_taps = len(tap_layers)
        self.tap_dim = tap_dim

        # Freeze ESM-2 completely
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()

        # Detect ESM hidden dim
        esm_dim = self.esm.config.hidden_size
        self.esm_dim = esm_dim

        # Per-tap projections (learned)
        self.tap_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(esm_dim, tap_dim),
                nn.LayerNorm(tap_dim))
            for _ in range(self.n_taps)])

        # Storage for hooked activations
        self._tap_cache = {}
        self._hooks = []

    def _install_hooks(self):
        """Install forward hooks on target layers."""
        self._remove_hooks()
        for layer_idx in self.tap_layers:
            layer = self.esm.esm.encoder.layer[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # ESM-2 layer output is (hidden_states, ...)
                    self._tap_cache[idx] = output[0].detach()
                return hook

            h = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._tap_cache.clear()

    def forward(self, input_ids, attention_mask=None):
        """Run frozen ESM-2, extract taps, project, stack.

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) optional mask

        Returns:
            tap_stack: (B, n_taps, tap_dim)
        """
        B, L = input_ids.shape

        # Install hooks
        self._install_hooks()

        # Forward through frozen ESM-2
        with torch.no_grad():
            esm_out = self.esm(input_ids, attention_mask=attention_mask)
            logits = esm_out.logits  # (B, L, vocab)
            final_hidden = esm_out.hidden_states[-1] if hasattr(esm_out, 'hidden_states') and esm_out.hidden_states else None

        # Collect taps and project
        tap_features = []
        for i, layer_idx in enumerate(self.tap_layers):
            h = self._tap_cache[layer_idx]  # (B, L, esm_dim)
            # Mean-pool over tokens (mask-aware)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = h.mean(1)
            projected = self.tap_projs[i](pooled)  # (B, tap_dim)
            tap_features.append(projected)

        # Stack: (B, n_taps, tap_dim)
        tap_stack = torch.stack(tap_features, dim=1)

        # Cleanup hooks
        self._remove_hooks()

        # Write to cache
        if self.parent is not None:
            self.parent.cache_set('tap_stack', tap_stack)
            self.parent.cache_set('esm_logits', logits)
            self.parent.cache_set('attention_mask', attention_mask)
            self.parent.cache_set('seq_len', L)
            # Store final hidden for token-level distillation
            if final_hidden is not None:
                self.parent.cache_set('esm_hidden_final', final_hidden)
            else:
                # Fallback: use last tap
                self.parent.cache_set('esm_hidden_final',
                                      self._tap_cache.get(self.tap_layers[-1]))

        return tap_stack

    def train(self, mode=True):
        """Override: ESM stays in eval always."""
        super().train(mode)
        self.esm.eval()
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# DISTILLATION HEAD — predict frozen model's logits from geometric features
# ═══════════════════════════════════════════════════════════════════════════════

class DistillationHead(TorchComponent):
    """Predict ESM-2 token logits from geometric features.

    Combines protein-level geometric feature with per-token hidden states
    from the final ESM-2 layer to predict the model's own token logits.

    The geometric feature conditions the per-token predictions:
    it tells each token what the overall protein structure looks like.

    Reads:  cache['geo_features']      — (B, geo_dim) protein-level
            cache['esm_hidden_final']  — (B, L, esm_dim) per-token
            cache['esm_logits']        — (B, L, vocab) target
            cache['attention_mask']    — (B, L)
    Writes: cache['pred_logits']       — (B, L, vocab) predicted
            cache['distill_loss']      — scalar
    """

    def __init__(self, name, geo_dim, esm_dim, vocab_size, temperature=2.0, **kwargs):
        super().__init__(name, **kwargs)
        self.temperature = temperature

        # Condition: broadcast geo features to token level
        self.geo_to_token = nn.Sequential(
            nn.Linear(geo_dim, esm_dim),
            nn.LayerNorm(esm_dim), nn.GELU())

        # Fuse conditioned geo + frozen hidden → logits
        self.predict = nn.Sequential(
            nn.Linear(esm_dim * 2, esm_dim),
            nn.GELU(), nn.LayerNorm(esm_dim),
            nn.Dropout(0.1),
            nn.Linear(esm_dim, vocab_size))

    def forward(self):
        geo_feat = self.parent.cache_get('geo_features')
        hidden = self.parent.cache_get('esm_hidden_final')
        target_logits = self.parent.cache_get('esm_logits')
        mask = self.parent.cache_get('attention_mask')

        B, L, D = hidden.shape

        # Broadcast geometric feature to every token
        geo_expanded = self.geo_to_token(geo_feat).unsqueeze(1).expand(-1, L, -1)

        # Fuse: per-token hidden + geometric context
        fused = torch.cat([hidden.detach(), geo_expanded], dim=-1)
        pred_logits = self.predict(fused)  # (B, L, vocab)

        # KL divergence loss (soft distillation)
        T = self.temperature
        log_pred = F.log_softmax(pred_logits / T, dim=-1)
        target_soft = F.softmax(target_logits.detach() / T, dim=-1)
        kl = F.kl_div(log_pred, target_soft, reduction='none').sum(-1)  # (B, L)

        if mask is not None:
            kl = (kl * mask.float()).sum() / mask.float().sum()
        else:
            kl = kl.mean()

        loss = kl * (T ** 2)

        if self.parent is not None:
            self.parent.cache_set('pred_logits', pred_logits)
            self.parent.cache_set('distill_loss', loss)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT EFFECT HEAD — fine-tune on fitness prediction
# ═══════════════════════════════════════════════════════════════════════════════

class VariantEffectHead(TorchComponent):
    """Predict variant fitness from geometric feature comparison.

    Given WT and mutant geometric features, predict fitness score.
    Concatenates [wt_feat, mut_feat, wt_feat - mut_feat, wt_feat * mut_feat].

    Reads:  cache['geo_features_wt']  — (B, geo_dim)
            cache['geo_features_mut'] — (B, geo_dim)
    Writes: cache['fitness_pred']     — (B, 1)
    """

    def __init__(self, name, geo_dim, **kwargs):
        super().__init__(name, **kwargs)
        self.head = nn.Sequential(
            nn.Linear(geo_dim * 4, geo_dim),
            nn.GELU(), nn.LayerNorm(geo_dim),
            nn.Dropout(0.1),
            nn.Linear(geo_dim, 1))

    def forward(self, wt_feat=None, mut_feat=None):
        if wt_feat is None and self.parent is not None:
            wt_feat = self.parent.cache_get('geo_features_wt')
        if mut_feat is None and self.parent is not None:
            mut_feat = self.parent.cache_get('geo_features_mut')

        combined = torch.cat([
            wt_feat, mut_feat,
            wt_feat - mut_feat,
            wt_feat * mut_feat], dim=-1)

        pred = self.head(combined)

        if self.parent is not None:
            self.parent.cache_set('fitness_pred', pred)
        return pred


# ═══════════════════════════════════════════════════════════════════════════════
# ESM-2 GEOMETRIC PIPELINE — the full router
# ═══════════════════════════════════════════════════════════════════════════════

class ESM2GeometricPipeline(BaseTower):
    """Geometric self-distillation on frozen ESM-2.

    Phase 1 — Self-distillation:
        Learns to read ESM-2's internal structure geometrically.
        No labels needed. The frozen model is the curriculum.

    Phase 2 — Variant effect fine-tuning:
        Predicts mutation fitness from geometric feature comparison.
        Uses MaveDB or ClinVar labels.

    Args:
        name:           Router name
        esm_model_name: HuggingFace model name for ESM-2
        tap_layers:     Which ESM-2 layers to tap (None = auto-space)
        tap_dim:        Projection dim per tap
        embed_dim:      Constellation embedding dimension
        n_anchors:      Constellation anchors
        n_comp:         Patchwork compartments
        d_comp:         Per-compartment dim
        gate_strategy:  CM gate strategy
        temperature:    Distillation temperature
    """

    def __init__(self, name, esm_model_name='facebook/esm2_t33_650M_UR50D',
                 tap_layers=None, tap_dim=256,
                 embed_dim=256, n_anchors=32, n_comp=8, d_comp=32,
                 gate_strategy='cm_gate', n_neighbors=3,
                 temperature=2.0, strict=False):
        super().__init__(name, strict=strict)

        # ── Load frozen ESM-2 ──
        from transformers import AutoModelForMaskedLM, AutoConfig
        config = AutoConfig.from_pretrained(esm_model_name)
        config.output_hidden_states = True
        esm_model = AutoModelForMaskedLM.from_pretrained(
            esm_model_name, config=config)

        n_layers = config.num_hidden_layers
        esm_dim = config.hidden_size
        vocab_size = config.vocab_size

        # Auto-space taps if not specified
        if tap_layers is None:
            if n_layers >= 24:
                tap_layers = [5, 11, 17, 23, n_layers - 1]
            elif n_layers >= 12:
                tap_layers = [2, 5, 8, n_layers - 1]
            else:
                tap_layers = [1, n_layers // 2, n_layers - 1]

        n_taps = len(tap_layers)
        pw_dim = n_comp * d_comp
        svd_feat_dim = 2 * n_taps + 2

        # Store config
        self.attach('config', {
            'esm_model_name': esm_model_name,
            'esm_dim': esm_dim, 'vocab_size': vocab_size,
            'n_layers': n_layers, 'tap_layers': tap_layers,
            'tap_dim': tap_dim, 'embed_dim': embed_dim,
            'n_anchors': n_anchors, 'n_comp': n_comp, 'd_comp': d_comp,
            'pw_dim': pw_dim, 'svd_feat_dim': svd_feat_dim,
            'gate_strategy': gate_strategy, 'temperature': temperature,
        })

        # ── Tap system (owns frozen ESM-2) ──
        self.attach('taps', ESM2TapSystem(
            'taps', esm_model, tap_layers, tap_dim))

        # ── Embedding projection: tap_stack → S^(embed_dim-1) ──
        self.attach('embed_proj', nn.Sequential(
            nn.Linear(tap_dim, embed_dim),
            nn.LayerNorm(embed_dim)))

        # ── Geometric pipeline stages ──
        self.attach('observe', ObserveSVDTokens(
            'observe', n_taps))

        self.attach('associate', AssociateConstellation(
            'associate', dim=embed_dim, n_anchors=n_anchors))

        self.attach('curate_gate', CurateCMGate(
            'curate_gate', n_anchors, embed_dim, n_comp,
            n_neighbors, gate_strategy))

        self.attach('curate_pw', CuratePatchwork(
            'curate_pw', n_anchors, n_comp, d_comp))

        self.attach('fuse', FuseGeometric(
            'fuse', svd_feat_dim, pw_dim, embed_dim))

        # ── Distillation head ──
        geo_dim = pw_dim + pw_dim + embed_dim
        self.attach('distill', DistillationHead(
            'distill', geo_dim, esm_dim, vocab_size, temperature))

        # ── Variant effect head (for phase 2) ──
        self.attach('variant', VariantEffectHead(
            'variant', geo_dim))

    @property
    def feature_dim(self):
        return self['fuse'].feature_dim

    def _geometric_forward(self, input_ids, attention_mask=None):
        """Run tap extraction + geometric pipeline. Returns geo features."""
        # Extract taps from frozen ESM-2
        tap_stack = self['taps'](input_ids, attention_mask)

        # SVD observation of multi-layer structure
        self['observe'](tap_stack)

        # Pool taps → sphere → triangulate
        pooled = tap_stack.mean(dim=1)
        emb = F.normalize(self['embed_proj'](pooled), dim=-1)
        self['associate'](emb)

        # Curate + interpret
        self['curate_gate']()
        self['curate_pw']()

        # Fuse
        features = self['fuse']()
        return features

    def forward_distill(self, input_ids, attention_mask=None):
        """Phase 1: Self-distillation forward pass.

        Args:
            input_ids: (B, L) amino acid token ids
            attention_mask: (B, L)

        Returns:
            loss: scalar distillation loss
            info: dict with diagnostics
        """
        features = self._geometric_forward(input_ids, attention_mask)
        loss = self['distill']()

        info = {
            'loss': loss.item(),
            'svd_S': self.cache_get('svd_S'),
            'gate_info': self.cache_get('gate_info'),
            'embedding': self.cache_get('embedding'),
            'feature_dim': features.shape[-1],
        }

        return loss, info

    def forward_variant(self, wt_ids, mut_ids,
                        wt_mask=None, mut_mask=None, fitness=None):
        """Phase 2: Variant effect prediction.

        Args:
            wt_ids:  (B, L) wild-type token ids
            mut_ids: (B, L) mutant token ids
            wt_mask, mut_mask: (B, L) attention masks
            fitness: (B,) ground truth fitness scores (optional)

        Returns:
            pred: (B, 1) predicted fitness
            loss: scalar MSE loss (if fitness provided)
            info: dict
        """
        # WT pass
        wt_feat = self._geometric_forward(wt_ids, wt_mask)
        self.cache_set('geo_features_wt', wt_feat)
        wt_gate_info = self.cache_get('gate_info')

        # Mut pass
        mut_feat = self._geometric_forward(mut_ids, mut_mask)
        self.cache_set('geo_features_mut', mut_feat)

        # Predict fitness
        pred = self['variant']()

        loss = None
        if fitness is not None:
            loss = F.mse_loss(pred.squeeze(-1), fitness)

        info = {
            'pred': pred.detach(),
            'wt_gate_info': wt_gate_info,
            'geo_diff_norm': (wt_feat - mut_feat).norm(dim=-1).mean().item(),
        }

        return pred, loss, info

    def forward(self, input_ids, attention_mask=None):
        """Default: distillation mode."""
        return self.forward_distill(input_ids, attention_mask)

    def get_diagnostics(self):
        return {
            'svd_S': self.cache_get('svd_S'),
            'svd_Vh': self.cache_get('svd_Vh'),
            'embedding': self.cache_get('embedding'),
            'gate_values': self.cache_get('gate_values'),
            'gate_info': self.cache_get('gate_info'),
            'patchwork': self.cache_get('patchwork'),
            'distill_loss': self.cache_get('distill_loss'),
        }

    def trainable_params(self):
        """Iterator over only the trainable (non-frozen) parameters."""
        frozen_ids = {id(p) for p in self['taps'].esm.parameters()}
        return (p for p in self.parameters() if id(p) not in frozen_ids)

    def make_optimizer(self, lr=3e-4, weight_decay=0.05):
        """Optimizer over trainable params only (excludes frozen ESM-2)."""
        params = list(self.trainable_params())
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self['taps'].esm.parameters())
        trainable = total - frozen
        print(f"ESM2GeometricPipeline Summary")
        print(f"  ESM-2 (frozen):  {frozen:>12,}")
        print(f"  Geometric:       {trainable:>12,}")
        print(f"  Total:           {total:>12,}")
        print(f"  Tap layers:      {self['config']['tap_layers']}")
        print(f"  Gate strategy:   {self['config']['gate_strategy']}")
        return trainable