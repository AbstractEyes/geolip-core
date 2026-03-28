# Claude Context: Ryan Spearman + Geometric Transformer
## Session: March 27-28, 2026

**Phil (AbstractPhil) — independent geometric deep learning researcher**
**HuggingFace: AbstractPhil | GitHub: AbstractEyes**
**GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (102GB VRAM)**

---

## CRITICAL RULES — READ BEFORE WRITING ANY CODE

1. **Never use global average pooling in geometric encoders** — destroys spatial structure, drops accuracy from ~70% to ~29%. Always flatten or use spatial statistics.
2. **FiLM on individual arms BEFORE composition, not after** — FiLM on interleaved E3 sequence HURTS (-0.057). FiLM on individual quaternion arms HELPS (+0.037).
3. **Never overwrite HF checkpoints without verifying provenance** — we lost good weights by pushing over existing checkpoints.
4. **The observer's constellation is NOT cosine similarity + softmax** — it's anchors on S^(d-1) with repulsion initialization, triangulation (1 - cos), CM determinant gating, patchwork compartments, bridge prediction. Use the real `ConstellationObserver` from `geolip_core.core.associate.constellation`.
5. **Use `network_to(device, strict=False)` not `.to(device)`** for anything inheriting from geofractal's BaseTower.
6. **Use `tqdm.auto` in Colab**, never `tqdm` with `file=sys.stdout`.
7. **MaveDB target sequences are DNA nucleotides, not protein** — need codon translation.
8. **Observer checkpoint path**: `prototype/v1_distill/epoch_9.pt` (NOT `esm2_geo_phase1_best.pt`).

---

## SYSTEM ARCHITECTURE

### The GeoLIP Pipeline (5 stages)

```
geolip_core/
├── core/
│   ├── input/           # SVDObserver, scatter, spectral
│   ├── associate/       # Constellation, Relay, FlowAttention
│   │   └── constellation.py  — THE real constellation:
│   │       Constellation (anchor primitive, repulsion-init on S^(d-1))
│   │       ConstellationAssociation (triangulation + soft assignment)
│   │       ConstellationCuration (patchwork + bridge)
│   │       ConstellationObserver (composed assoc + curation)
│   ├── curate/          # AnchorGate (CM det), Patchwork, MagnitudeFlow
│   ├── align/           # ProcrustesAlignment, whitening
│   └── distinguish/     # Losses, task heads
├── pipeline/
│   ├── observer.py      # TorchComponent, BaseTower, Input, Curation, Distinction
│   ├── layer.py         # ConstellationLayer (one depth) — stub
│   ├── backbone.py      # GeometricBackbone (multi-depth) — stub
│   └── geolip.py        # GeoLIP composition container — stub
└── example/             # Working models
```

### Key Interfaces

```python
# TorchComponent — standalone nn.Module with identity + lifecycle
class MyStage(TorchComponent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

# BaseTower — composition container
tower = BaseTower('name')
tower.attach('stage_name', module)  # nn.Module
tower['stage_name']                  # access
tower.has('stage_name')              # check
tower.cache_set('key', tensor)       # ephemeral
tower.network_to(device=device, strict=False)  # device placement
```

### ConstellationObserver Interface

```python
obs = ConstellationObserver(dim=256, n_anchors=32, n_comp=8, d_comp=32)
result = obs.observe(emb)  # emb: (B, dim) on S^(dim-1)
# Returns dict:
#   'embedding':      (B, dim)
#   'features':       (B, feature_dim)  = assignment + patchwork + embedding
#   'triangulation':  (B, A)            = 1 - cos(emb, anchors)
#   'cos_to_anchors': (B, A)
#   'nearest':        (B,)
#   'assignment':     (B, A)            = softmax(cos / temp)
#   'patchwork':      (B, pw_dim)
#   'bridge':         (B, A)
```

---

## RYAN SPEARMAN — Variant Effect Prediction

### What It Is
Prediction heads on frozen ESM-2 (650M) + GeoLIP observer that predict mutation fitness effects. Named for Ryan Spears (1987-2013).

### HuggingFace Repos
- **Observer**: `AbstractPhil/geolip-esm2_t33_650M_UR50D`
  - Checkpoint: `prototype/v1_distill/epoch_9.pt`
- **Features + Heads**: `AbstractPhil/ryan-spearman-prepared-features`
  - `shards/shard_0000.pt` through `shard_0009.pt` — 22,020 pre-extracted features
  - `heads/GeoQuat_epoch100.pt` — 7.4M params, ρ=0.993 training
  - `heads/E3_epoch100.pt` — 1.6M params, ρ=0.980 training
  - `heads/E3_Baseline_best.pt` — same as E3_epoch100 (overwritten)
  - `heads/Procrustes_matched_epoch200.pt` — 511K params, ρ=0.962 training

### Training Data
5 MaveDB proteins, 22,020 variants total:
- BRCA1 (1,222), PTEN (10,469), SUMO1 (1,919), TPK1 (5,408), UBE2I (3,002)

### ProteinGym Benchmark Results (84 unseen assays, v0.1)

| Head | Mean ρ | Median ρ | P25 | P75 | Wins vs GeoQuat |
|---|---|---|---|---|---|
| **Procrustes matched** | **0.309** | **0.302** | 0.180 | 0.416 | **76/84 (90%)** |
| GeoQuat E100 | 0.277 | 0.276 | 0.139 | 0.371 | — |
| Procrustes E200 (mismatched) | 0.273 | 0.252 | 0.132 | 0.380 | — |
| E3 E100 | 0.245 | 0.236 | 0.136 | 0.321 | — |

Published comparison: ESM-2 zero-shot ~0.45-0.52, AlphaMissense 0.514, S3F-MSA ~0.58

### Key Findings
1. **Procrustes matched wins 76/84 assays** — Bertenstein principle confirmed
2. **GeoQuat > E3 by 13%** — geometric FiLM conditioning transfers, raw MHA doesn't
3. **Mismatched Procrustes WORSE than GeoQuat solo** — matched-strength experts essential
4. **Training diversity is the bottleneck** — 5 proteins → 0.309. Architecture works.
5. **Beta-lactamase: Procrustes hits 0.550** approaching SOTA from zero training data
6. **MHA memorizes without constraint** — quaternion algebra acts as structural regularizer
7. **Hamilton product forces complementary representations** — arms can't independently memorize
8. **Early stopping vs full training**: 32-epoch (ρ=0.916) got 0.278 on PG; 100-epoch (ρ=0.993) got 0.277 — nearly identical generalization despite 0.077 training gap

### Architecture Details

**GeoQuaternionHead (7.4M params)**:
- GeoContextEncoder: 3 streams (constellation, structural, opinion) → 128-d FiLM context
- 4 arms with FiLM conditioning between MHA blocks:
  - w-arm: masked layers (conservation)
  - i-arm: WT layers (context)
  - j-arm: MUT layers (consequence)
  - k-arm: WT-MUT layers (displacement)
- Each arm → 64-d quaternion component
- Hamilton product with learned rotation → 256-d composed
- Concatenate 128-d geo context → 384-d → MLP → scalar fitness

**E3_Baseline (1.6M params)**:
- 3-condition cross-attention (WT, masked, MUT interleaved as 99 tokens)
- 2 MHA blocks, no geometric features, pure ESM-2 layers
- repr_dim = 832 (256*3 + 64)

**ProcrustesAligner (511K params)**:
- Projects both expert representations to common 256-d
- Cayley orthogonal rotation: Q = (I-A)(I+A)^(-1), skew-symmetric A
- Newton-Schulz whitening during training
- MLP predicts fitness from aligned 512-d concatenation

### Feature Extraction Per Variant
44 feature fields:
- `esm_at_pos_wt/masked/mut`: (33, 1280) ESM-2 layers at mutation site × 3 conditions
- `esm_logits_masked`, `obs_logits_masked`: (33,) logits at position
- `geo_wt/mut`: (768,) mean-pooled geometric features
- `gate_wt/mut`: (32,) anchor gate activations
- `patchwork_wt/mut`: (256,) compartment features
- `embed_wt/mut`: (256,) hypersphere embedding
- `svd_S_wt/mut`: (5,) singular value spectrum
- `cos_to_anchors_wt/mut`: (32,) cosine to constellation anchors
- `tri_distances_wt/mut`: (32,) triangulation distances
- `soft_assignment_wt/mut`: (32,) soft anchor assignment
- `tri_gated_wt/mut`: (32,) CM-gated triangulation

**Note**: All geometric features are PROTEIN-LEVEL (mean-pooled), not position-specific. The observer's per-position tap stack was never fed to the heads. This is a known limitation and the next improvement target.

---

## GEOMETRIC TRANSFORMER — Prototype

### File: `geometric_transformer.py`

**Status**: Prototype passes self-test, 4.18M params (small config), 283.5M params (ESM-2 scale = 43.6% overhead on 650M base).

### Architecture Per Layer

```
hidden_states → ManifoldProjection → emb on S^(manifold_dim - 1) per position
             → ConstellationObserver.observe() → full observation dict
             → PositionGeometricContext → FiLM context (B, L, context_dim)
             → Stream A: ContentAttention (standard MHA, no geometry)
             → Stream B: GeometricAttention (FiLM on Q,K from constellation, V pure)
             → CayleyOrthogonal (align B basis → A basis)
             → QuaternionCompose (w=content, i=aligned_geo, j=disagree, k=agree)
             → decode + gated residual → x_out
```

### Components

| Component | Source | Status |
|---|---|---|
| `ConstellationObserver` | `geolip_core.core.associate.constellation` | Real, imported |
| `ConstellationAssociation` | `geolip_core.core.associate.constellation` | Real, imported |
| `ConstellationCuration` | `geolip_core.core.associate.constellation` | Real, imported |
| `Constellation` | `geolip_core.core.associate.constellation` | Real, imported |
| `TorchComponent` / `BaseTower` | `geolip_core.pipeline.observer` | Real, imported |
| `FiLMLayer` | Ryan Spearman (proven) | New, tested |
| `CayleyOrthogonal` | Ryan Spearman (proven) | New, tested |
| `QuaternionCompose` | Ryan Spearman (proven) | New, tested |
| `ManifoldProjection` | New (Input stage) | New |
| `PositionGeometricContext` | New (Curation stage) | New |
| `GeometricAttention` | New (FiLM Q,K) | New |
| `ContentAttention` | New (standard MHA) | New |
| `GeometricTransformerLayer` | New (BaseTower composition) | New |
| `GeometricTransformer` | New (layer stack) | New |

### Layer Returns Full Geometric State

```python
x_out, geo_state = layer(x)
# geo_state dict keys:
#   embedding:      (B, L, manifold_dim)   per-position on S^(d-1)
#   geo_ctx:        (B, L, context_dim)    compressed FiLM vector
#   triangulation:  (B, L, A)             cosine distances to anchors
#   cos_to_anchors: (B, L, A)             raw cosine similarities
#   assignment:     (B, L, A)             soft assignment
#   nearest:        (B, L)                nearest anchor index
#   patchwork:      (B, L, pw_dim)        compartment features
#   bridge:         (B, L, A)             patchwork's assignment estimate
#   content:        (B, L, D)             Stream A output
#   geometric:      (B, L, D)             Stream B output (pre-rotation)
#   composed:       (B, L, 4*quat_dim)    raw quaternion composition
```

### Factories

```python
geo_transformer_esm2('name', n_layers=6)    # d=1280, 16 heads, 283.5M params
geo_transformer_small('name', n_layers=4)   # d=256, 8 heads, ~4.2M params
geo_transformer_vision('name', n_layers=4)  # d=384, 8 heads, for scatter/SVD patches
```

---

## GEOMETRIC CONSTANTS (validated across 17+ models)

- **CV pentachoron band**: 0.20–0.23 (universal attractor on S^(d-1))
- **Binding/separation constant**: 0.29154 (complement 0.70846)
- **Cross-modal QK eigenvalue lock**: 0.500 (universal)
- **Cayley rotation convergence**: ‖R-I‖ ≈ 4.1 at 200 epochs in SO(256)

---

## PENDING WORK

1. **Train heads on 30-40 diverse ProteinGym proteins** — the clear next step for Ryan Spearman
2. **Extract per-position observer tap features** — `tap_stack_at_pos` for local geometric conditioning
3. **Run ProteinGym v1.3** (217 assays) — current results on v0.1 (87 assays)
4. **Integrate geometric transformer into geolip-core** as `example/geometric_transformer.py`
5. **Train geometric transformer on scatter/SVD vision pipeline** — validate constellation routing on images
6. **Clinical variant classification** (ClinVar) with Ryan Spearman
7. **Complete pipeline stubs**: `pipeline/layer.py`, `pipeline/backbone.py`, `pipeline/geolip.py`

---

## OUTPUT FILES FROM THIS SESSION

- `/mnt/user-data/outputs/geometric_transformer.py` — Full prototype, self-test passes
- `/mnt/user-data/outputs/eval_proteingym.py` — Multi-head ProteinGym benchmark (4 heads)
- `/mnt/user-data/outputs/train_full_epochs.py` — GeoQuat 100 + Procrustes 200 trainer
- `/mnt/user-data/outputs/train_matched_experts.py` — E3 100 + matched Procrustes 200
- `/mnt/user-data/outputs/ryan_spearman_article.md` — Complete paper with all results

## TRANSCRIPTS

- `/mnt/transcripts/2026-03-28-04-37-51-ryan-spearman-benchmark-procrustes-full-session.txt`
- `/mnt/transcripts/2026-03-27-20-37-40-ryan-spearman-variant-effect-full-session.txt`
- `/mnt/transcripts/2026-03-27-01-09-24-esm2-geometric-observer-full-session.txt`
- `/mnt/transcripts/2026-03-26-10-41-11-geolip-core-refactor-esm2-observer.txt`

---

## TOTAL COMPUTE

~$16, ~16 hours on single Blackwell GPU. Phase 1 observer distillation (13h) is the bottleneck. Head training is 24 minutes total. ProteinGym benchmark is 76 minutes.