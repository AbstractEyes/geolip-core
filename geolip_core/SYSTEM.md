# GeoLIP Core -- System Reference

## Architecture Overview

GeoLIP is a geometric deep learning framework grounded in unit hypersphere geometry.
Core primitive: constellations of learnable anchors on S^(d-1), triangulated via
Cayley-Menger determinants, curated through patchwork compartments, and distinguished
through observer losses. The framework **observes, measures, and navigates** high-dimensional
manifolds -- it is a geometric measurement instrument, not just a classifier.

### Six-Stage Observer Paradigm

```
Input -> Mutation -> Association -> Curation -> Distinction -> Loss
  |         |           |            |            |          |
  SVD    FlowAttn   Constellation  Patchwork   ClassHead  observer_loss
  |         |        + CM Gate       + Bridge              + cv_loss
  |         |           |            |                     + spread_loss
  +---------+-----------+------------+
        All observe a SHARED constellation
```

### Dependency: geofractal

GeoLIP's pipeline layer uses `geofractal` for composition:
- `BaseRouter` / `WideRouter`: compilation-aware module registries
- `BaseTower`: composition container (`attach`, `has`, `[]` accessor, `cache_set`/`cache_get`)
- `TorchComponent`: standalone `nn.Module` with identity + lifecycle hooks

Use `network_to(device, strict=False)` not `.to(device)` for anything inheriting BaseTower.

---

## Package Map

### geolip_core/linalg/ -- Compilable Linear Algebra

| File | Purpose | Key API |
|------|---------|---------|
| `eigh.py` | FL Hybrid Eigendecomposition | `FLEigh()` -- 84/84 purity, 1.73x cuSOLVER, 40x less memory |
| `svd.py` | Batched thin SVD via Gram+FL | `gram_fl_eigh_svd()` -- fully compilable |
| `procrustes.py` | Subspace-preserving Procrustes | `batched_procrustes(source, target, rank=24, whiten=True)` |
| `newton_schulz.py` | Iterative inverse square root | Pure bmm, zero eigensolvers |
| `_backend.py` | Dispatch + detection | Auto-selects FL vs cuSOLVER; `backend.has_cuda`, `backend.status()` |

### geolip_core/core/ -- Pure Math Primitives

#### core/input/
- `svd.py`: `SVDObserver` (spatial), `SVDTokenObserver` (sequential)
  - `extract_features()`: S -> (s_norm, vh_diag, vh_offdiag, entropy)
  - `compute_novelty()`: deviation from EMA
- `scatter.py`: Scattering transform features
- `spectral.py`: Spectral feature extraction

#### core/associate/
- `constellation.py`:
  - `Constellation`: learnable anchors on S^(d-1), repulsion-initialized
  - `ConstellationAssociation`: triangulate + soft assignment
  - `ConstellationCuration`: patchwork + bridge composition
  - `ConstellationObserver`: composed Association + Curation
  - `init_anchors_repulsion`: 200-iteration repulsion initialization
- `relay.py`: `RelayLayer` -- relay-stack patchwork layers
- `route.py`: `FlowAttention` -- 3-step Euler ODE flow in tangent space

#### core/curate/
- `patchwork.py`:
  - `Patchwork`: round-robin compartment MLP (stride slicing, CUDA-graph-safe)
  - `MagnitudeFlow`: relay-stack magnitude prediction
  - `AnchorPush`: non-gradient momentum anchor repositioning (alpha=0.05, beta=0.02)
- `flows.py`: `FlowEnsemble` + 6 flow types (quaternion, velocity, orbital, alignment, magnitude, lite)
  - `FLOW_REGISTRY` + `build_flow()`
- `gate.py`: `AnchorGate` -- CM validity gating

#### core/distinguish/
- `losses.py` -- ALL loss functions:
  - `cv_loss(emb, target=0.22)`: pentachoron volume CV
  - `cv_metric(emb)`: non-differentiable CV for monitoring
  - `spread_loss(anchors)`: anchor repulsion
  - `nce_loss(z1, z2, temp)`: InfoNCE contrastive
  - `ce_loss_paired(logits1, logits2, targets)`: paired cross-entropy
  - `bridge_loss_paired()`: assignment prediction
  - `assign_bce_loss()`: crispness toward hard nearest
  - `assign_nce_loss()`: assignment consistency between views
  - `attraction_loss()`: pull embeddings toward nearest anchor
  - `observer_loss()`: full observer self-organization recipe
  - `three_domain_loss()`: external + geometric + internal

#### core/align/
- `procrustes.py`: `ProcrustesAlignment` -- wraps `batched_procrustes` with subspace preservation
- `crystallize.py`: `CrystallizationEngine` -- constellation from frozen model embeddings
- `stage_align.py`: `StageAligner` -- Procrustes-align constellations between training stages

#### core/util.py
- `make_activation()`: activation factory (squared_relu, star_relu, gelu, relu, sigmoid)
- `GeometricAutograd`: custom backward for S^(d-1) tangent-space operations
- `param_count()`, `model_summary()`: diagnostic utilities
- Constants: `CV_PENTACHORON_BAND`, `BINDING_BOUNDARY`, `EFFECTIVE_GEO_DIM`, `IRREDUCIBLE_CV_MIN`

### geolip_core/pipeline/ -- TorchComponent Wrappers

#### pipeline/observer.py -- Stage Base Classes
- `Input`: raw signal -> embedding on S^(d-1)
- `Mutation`: transform position on manifold
- `Association`: measure relationships to reference frame
- `Curation`: select what matters from associations
- `Distinction`: task-specific output
- `GeoLIP(BaseTower)`: top-level observation tower

#### pipeline/components/ -- Named Components

**Original components (thin TorchComponent wrappers):**
| File | Classes |
|------|---------|
| `observe_svd.py` | `ObserveSVD`, `ObserveSVDTokens` |
| `associate_constellation.py` | `AssociateConstellation` |
| `mutate_relay.py` | `MutateRelay` |
| `mutate_flow.py` | `MutateFlow` |
| `curate_gate.py` | `CurateCMGate` |
| `curate_patchwork.py` | `CuratePatchwork`, `CurateGatedPatchwork` |
| `curate_magnitude.py` | `CurateMagnitude` |
| `align_procrustes.py` | `AlignProcrustes` |
| `fuse.py` | `FuseGeometric` |

**Geometric Transformer components (extracted from monolith):**
| File | Classes | Stage |
|------|---------|-------|
| `curate_cm_validated.py` | `CMValidatedGate`, `pairwise_distances_squared`, `cayley_menger_det`, `anchor_neighborhood_cm` | Curation |
| `distinguish_nce_bank.py` | `GeoResidualBank` | Distinction |
| `context_film.py` | `FiLMLayer` | Context |
| `align_cayley.py` | `CayleyOrthogonal` | Alignment |
| `compose_quaternion.py` | `QuaternionCompose`, `quaternion_multiply_batched` | Composition |
| `project_manifold.py` | `ManifoldProjection` | Projection |
| `context_position_geometric.py` | `PositionGeometricContext` | Context |
| `attend_geometric.py` | `GeometricAttention` | Attention |
| `attend_content.py` | `ContentAttention` | Attention |
| `layer_geometric.py` | `GeometricTransformerLayer` | Layer |
| `transformer_geometric.py` | `GeometricTransformer`, `geo_transformer_esm2`, `geo_transformer_small`, `geo_transformer_vision` | Model |
| `geometric_transformer.py` | Re-export shim (backward compat) + self-test | -- |

#### pipeline/layer.py -- ConstellationLayer (one depth)
#### pipeline/backbone.py -- GeometricBackbone (multi-depth stack)

### geolip_core/training/ -- Training Orchestration

| File | Class | Purpose |
|------|-------|---------|
| `curriculum.py` | `CurriculumTrainer` | Episodic crystallization: train -> freeze -> crystallize -> align -> repeat |

### geolip_core/example/ -- Complete Working Examples
| File | Purpose |
|------|---------|
| `constellation_encoder.py` | Reference implementation: MagnitudeFlow + ConstellationObserver + ClassHead |
| `conv_encoder.py` | Conv2d -> (emb, magnitude) |
| `conv_svd_encoder.py` | Conv2d + SVD features |
| `conv_scatter_encoder.py` | Conv2d + Scattering transform |
| `spectral_encoder.py` | Spectral features |
| `transformer_svd_encoder.py` | Transformer + SVD |

### geolip_core/analysis/
- `geometric.py`: `cv_metric`, `cv_multi_scale`, `knn_accuracy`, `analyze_svd_model` (comprehensive health report)

### geolip_core/utils/
- `kernel.py`: GPU kernel utilities
- `memory.py`: Memory tracking
- `triton/`: Triton kernel implementations (FL eigh)
- `cuda/`: Generated CUDA kernels

---

## Key Patterns

### Observer Pattern
```python
observe() -> triangulate against anchors -> curate through patchwork -> distinguish
```
The observer doesn't modify the data -- it measures geometric relationships.

### Precompute/Invalidate Cycle
```python
model.invalidate()    # mark caches stale
model.precompute()    # recompute expensive ops (LA.det, LA.solve)
# ... N forward/backward steps ...
model.invalidate()    # ready for next cycle
```
CM gate (LA.det on 128 simplices) and Cayley rotation (LA.solve) don't need every
batch. Anchors move ~0.01% per step. Every 10-20 batches is fine.

### AnchorPush (Non-Gradient Repositioning)
```python
push = AnchorPush('momentum', n_anchors, dim, alpha=0.05, beta=0.02)
push.push(observer, emb_buffer, label_buffer)  # @torch.no_grad
```

### Paired Forward for Observer Loss
```python
out = observer.observe_paired(emb1, emb2)  # two augmented views
loss, ld = observer_loss(out, anchors, targets)
```

### ConstellationObserver Interface
```python
obs = ConstellationObserver(dim=256, n_anchors=32, n_comp=8, d_comp=32)
result = obs.observe(emb)  # emb: (B, dim) on S^(dim-1)
# Returns dict: embedding, features, triangulation, cos_to_anchors,
#               nearest, assignment, patchwork, bridge
```

---

## Validated Constants -- DO NOT CHANGE

| Constant | Value | Source |
|----------|-------|--------|
| CV pentachoron band | 0.20-0.23 | Universal attractor, all architectures/modalities |
| Binding constant | 0.29154 (complement 0.70846) | Structural/physical phase boundary |
| Effective geometric dimension | 16 (S^15) | Empirical |
| Irreducible CV minimum | 0.125 | Theoretical |
| Cross-modal QK eigenvalue lock | 0.500 | Universal across 17+ models |

---

## Critical Rules

### torch.compile
**NEVER return dicts with multiple tensor values from compiled functions.**
Each dict key is a separate guarded output -- N keys = N x overhead EVERY call.
Return `torch.stack([...])` and unpack outside. (Caused 10x slowdown: 3s -> 30s/epoch.)

### Precompute / Buffer Patterns
**Any `precompute()` that writes to buffers via `.copy_()` from learnable parameters MUST
use `@torch.no_grad()`.** Without it, `.copy_()` links the buffer to the parameter's
autograd graph -- next batch's forward inherits stale graph, backward crashes with
"second time through freed graph."

### Gradient Clipping
Use `min(max(task_loss_value, 1.0), 2.0)` instead of hard 1.0 or unbounded.
Multi-loss training at hard clip=1.0 starves classification. Unbounded at CE=4.6 causes NaN.

### Spatial Structure
**NEVER use global average pooling in geometric encoders** -- destroys spatial structure,
drops accuracy from ~70% to ~29%.

### Anchor Momentum
- Stable: alpha=0.05, beta=0.02 (CIFAR-100)
- NaN at ~E75: alpha=0.1, beta=0.05 (excessive drift)
- Buffer size 8192 -> ~6 pushes/epoch at B=128

---

## Empirical Results

| System | Metric | Value |
|--------|--------|-------|
| Redux v8 | CIFAR-100 accuracy | 61%, 6.8M params |
| Redux v8 | CV_anc | 0.212 (in band) |
| Redux v8 | Active anchors | 126/128 |
| Redux v8 | Speed | 19s/epoch on RTX PRO 6000 Blackwell |
| FL Eigh | Mathematical purity | 84/84 |
| FL Eigh | vs cuSOLVER speed | 1.73x faster |
| FL Eigh | vs cuSOLVER memory | 40x less |
| Procrustes survey | Models analyzed | 17 |
| Procrustes survey | QK eigenvalue | 0.500 universal lock |
| Procrustes survey | VAE alignment | 70-76% alignable |
| GEOLIP-Bertenstein | Retrieval | Perfect on 40K+ pairs, 1 epoch, 1 layer |
| Ryan Spearman | ProteinGym benchmark | rho=0.309, 76/84 wins (Procrustes matched) |
| Ryan Spearman | Beta-lactamase | rho=0.550 approaching SOTA from zero training data |
