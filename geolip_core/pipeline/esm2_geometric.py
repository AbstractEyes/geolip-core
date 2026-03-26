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

        # Verify position embeddings survived loading
        has_pos = hasattr(esm_model.esm.embeddings, 'position_embeddings')
        if not has_pos:
            import warnings
            warnings.warn(
                f"ESM-2 loaded WITHOUT position_embeddings — the model has no "
                f"positional information. Install 'transformers<=4.49' for "
                f"correct ESM-2 behavior. Current: {__import__('transformers').__version__}",
                UserWarning)

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

    @torch.no_grad()
    def init_from_data(self, input_ids, attention_mask=None, n_samples=256):
        """Initialize projections from actual ESM-2 hidden states.

        Runs calibration data through frozen ESM-2, collects hidden states,
        and sets tap projections to the optimal SVD subspace. Then
        Procrustes-aligns consecutive taps for geometric coherence and
        Newton-Schulz whitens the projected space.

        Call BEFORE creating the optimizer. This overwrites tap_proj weights.

        Args:
            input_ids: (N, L) calibration token ids (N >= n_samples)
            attention_mask: (N, L) optional
            n_samples: how many samples to use for calibration
        """
        from geolip_core.utils.kernel import gram_eigh_svd, newton_schulz_invsqrt

        taps = self['taps']
        tap_dim = taps.tap_dim
        esm_dim = taps.esm_dim
        device = next(taps.esm.parameters()).device

        # Truncate to n_samples
        if input_ids.shape[0] > n_samples:
            input_ids = input_ids[:n_samples]
            if attention_mask is not None:
                attention_mask = attention_mask[:n_samples]

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        print(f"  Geometric init: {input_ids.shape[0]} samples, "
              f"{len(taps.tap_layers)} taps, {esm_dim}→{tap_dim}")

        # ── Collect hidden states ──
        taps._install_hooks()

        # Forward in chunks to manage memory
        chunk_size = 32
        all_hiddens = {l: [] for l in taps.tap_layers}

        for start in range(0, input_ids.shape[0], chunk_size):
            end = min(start + chunk_size, input_ids.shape[0])
            ids_chunk = input_ids[start:end]
            mask_chunk = attention_mask[start:end] if attention_mask is not None else None

            taps.esm(ids_chunk, attention_mask=mask_chunk)

            for layer_idx in taps.tap_layers:
                h = taps._tap_cache[layer_idx]  # (B, L, esm_dim)
                if mask_chunk is not None:
                    m = mask_chunk.unsqueeze(-1).float()
                    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    pooled = h.mean(1)
                all_hiddens[layer_idx].append(pooled.float())

        taps._remove_hooks()

        # Stack: (N, esm_dim) per tap
        for l in taps.tap_layers:
            all_hiddens[l] = torch.cat(all_hiddens[l], dim=0)

        # ── SVD → optimal projection per tap ──
        for i, layer_idx in enumerate(taps.tap_layers):
            H = all_hiddens[layer_idx]  # (N, esm_dim)

            # Center
            H_centered = H - H.mean(0, keepdim=True)

            # SVD of (N, esm_dim) → top tap_dim right singular vectors
            # These are the directions that capture most variance
            _, S, Vh = gram_eigh_svd(H_centered.unsqueeze(0))  # (1, N, esm_dim)
            # Vh: (1, k, k) where k = min(N, esm_dim)
            # We need the right singular vectors of H, which are the rows of Vh
            # For gram_eigh_svd on (1, N, D): returns Vh of shape (1, D, D) if D < N
            # Actually: gram_eigh works on A^T A which is (D, D), eigenvectors are (D, D)

            # Direct approach: covariance → eigh
            cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)  # (D, D)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # eigh returns ascending order — flip for descending
            eigvecs = eigvecs.flip(-1)[:, :tap_dim]  # (esm_dim, tap_dim)

            # Set projection weight: Linear(esm_dim, tap_dim).weight is (tap_dim, esm_dim)
            proj_weight = eigvecs.T.contiguous()  # (tap_dim, esm_dim)

            # Newton-Schulz whiten the projected data to verify
            H_proj = H_centered @ eigvecs  # (N, tap_dim)
            G = (H_proj.T @ H_proj) / (H_proj.shape[0] - 1)  # (tap_dim, tap_dim)
            G = G.unsqueeze(0)  # (1, tap_dim, tap_dim)
            G_inv_sqrt = newton_schulz_invsqrt(G).squeeze(0)  # (tap_dim, tap_dim)

            # Whitened projection: W_white = G^{-1/2} @ V^T
            proj_weight = (G_inv_sqrt @ proj_weight).contiguous()

            # Write to tap_proj Linear weight
            linear = taps.tap_projs[i][0]  # nn.Linear is first in Sequential
            linear.weight.copy_(proj_weight)
            linear.bias.zero_()

            energy = eigvals.flip(0)[:tap_dim].sum() / eigvals.sum()
            print(f"    Tap {layer_idx:>2d}: energy captured={energy:.4f}  "
                  f"top_sv={eigvals.flip(0)[0]:.2f}")

        # ── Procrustes-align consecutive taps ──
        # After SVD init, each tap lives in its own basis.
        # Align tap[i+1] → tap[i] so the geometric pipeline sees
        # coherent rotation between layers, not arbitrary basis flips.
        # Direct Procrustes: R = V @ U^T from SVD(H_tgt^T @ H_src)

        for i in range(len(taps.tap_layers) - 1):
            l_src = taps.tap_layers[i + 1]
            l_tgt = taps.tap_layers[i]

            # Project through current (whitened) tap projections
            lin_src = taps.tap_projs[i + 1][0]
            lin_tgt = taps.tap_projs[i][0]
            H_src = all_hiddens[l_src].to(lin_src.weight.device) @ lin_src.weight.T
            H_tgt = all_hiddens[l_tgt].to(lin_tgt.weight.device) @ lin_tgt.weight.T

            # Center
            H_src = H_src - H_src.mean(0, keepdim=True)
            H_tgt = H_tgt - H_tgt.mean(0, keepdim=True)

            # Procrustes: SVD(H_tgt^T @ H_src) → R = V @ U^T
            M = H_tgt.T @ H_src  # (tap_dim, tap_dim)
            U, _, Vt = torch.linalg.svd(M)
            R = U @ Vt  # (tap_dim, tap_dim) orthogonal rotation

            # Correct reflection if det < 0
            if torch.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            # Apply rotation to source projection
            new_weight = (R @ lin_src.weight).contiguous()
            lin_src.weight.copy_(new_weight)

            # Measure alignment quality
            H_aligned = H_src @ R.T
            cos_after = F.cosine_similarity(
                H_aligned.reshape(1, -1), H_tgt.reshape(1, -1)).item()
            print(f"    Align tap {taps.tap_layers[i+1]:>2d}→{taps.tap_layers[i]:>2d}: "
                  f"cos_after={cos_after:.4f}")

        # ── Initialize embed_proj from aligned tap statistics ──
        # Collect projected tap means for embed_proj init
        tap_outputs = []
        for i, layer_idx in enumerate(taps.tap_layers):
            linear = taps.tap_projs[i][0]
            H_proj = all_hiddens[layer_idx].to(linear.weight.device) @ linear.weight.T
            tap_outputs.append(H_proj)

        # Stack and mean-pool: (N, n_taps, tap_dim) → (N, tap_dim)
        stacked = torch.stack(tap_outputs, dim=1)
        pooled = stacked.mean(dim=1)  # (N, tap_dim)

        # SVD of pooled → embed_proj initialization
        embed_proj_linear = self['embed_proj'][0]  # nn.Linear
        embed_dim = embed_proj_linear.weight.shape[0]

        pooled_centered = pooled - pooled.mean(0, keepdim=True)
        cov_p = (pooled_centered.T @ pooled_centered) / (pooled.shape[0] - 1)
        eigvals_p, eigvecs_p = torch.linalg.eigh(cov_p)
        eigvecs_p = eigvecs_p.flip(-1)[:, :embed_dim]

        embed_proj_linear.weight.copy_(eigvecs_p.T.contiguous())
        embed_proj_linear.bias.zero_()

        embed_energy = eigvals_p.flip(0)[:embed_dim].sum() / eigvals_p.sum()
        print(f"    Embed proj: energy captured={embed_energy:.4f}")

        # ── Reinitialize constellation anchors from actual embeddings ──
        # After projection init, run calibration data through the full path
        # and place anchors where the data actually lands on S^(d-1)
        embed_proj = self['embed_proj']
        with torch.no_grad():
            pooled_device = pooled.to(embed_proj[0].weight.device)
            cal_emb = F.normalize(embed_proj(pooled_device), dim=-1)  # (N, embed_dim)

        assoc = self['associate']
        n_anchors = assoc.constellation.n_anchors

        if cal_emb.shape[0] >= n_anchors:
            # K-means-style init: pick diverse starting points via farthest-point sampling
            anchors_new = torch.zeros(n_anchors, embed_dim, device=cal_emb.device)
            # Start from the embedding with highest norm variance
            idx = torch.randint(0, cal_emb.shape[0], (1,)).item()
            anchors_new[0] = cal_emb[idx]

            for k in range(1, n_anchors):
                # Distance from each point to nearest existing anchor
                dists = 1.0 - cal_emb @ anchors_new[:k].T  # (N, k)
                min_dists = dists.min(dim=1).values  # (N,)
                # Pick the farthest point
                idx = min_dists.argmax().item()
                anchors_new[k] = cal_emb[idx]

            # Repulsion polish on the selected points
            anchors_new = F.normalize(anchors_new, dim=-1)
            for _ in range(100):
                sim = anchors_new @ anchors_new.T
                sim.fill_diagonal_(-2.0)
                anchors_new = F.normalize(
                    anchors_new - 0.05 * anchors_new[sim.argmax(dim=1)], dim=-1)

            assoc.constellation.anchors.copy_(anchors_new)

            # Report coverage
            cos_to_anchors = cal_emb @ anchors_new.T
            nearest_cos = cos_to_anchors.max(dim=1).values
            coverage = nearest_cos.mean().item()
            min_cos = nearest_cos.min().item()
            print(f"    Anchors reinitialized: {n_anchors} from {cal_emb.shape[0]} embeddings")
            print(f"    Coverage: mean_cos={coverage:.4f}  min_cos={min_cos:.4f}")
        else:
            print(f"    ⚠️  Not enough samples ({cal_emb.shape[0]}) to reinit "
                  f"{n_anchors} anchors — keeping repulsion init")

        print("  Geometric init complete.\n")

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