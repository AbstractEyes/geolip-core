# GeoLIP Core — Claude Code Task Plan

## Overview
Three tasks in priority order:
1. Component directory reorganization (untangle the monolith)
2. Curriculum Observer prototype (episodic crystallization pipeline)
3. System documentation (comprehensive MD covering all utilities)

---

## Task 1: Component Directory Reorganization

### Problem
`pipeline/components/geometric_transformer.py` is 1444 lines with 11 classes crammed in.
The rest of `pipeline/components/` is properly factored (30-80 lines each, one concern per file).

### Current `pipeline/components/` Contents
```
__init__.py              (20 lines) — imports
align_procrustes.py      (30 lines) — AlignProcrustes
associate_constellation.py (52 lines) — AssociateConstellation
curate_gate.py           (50 lines) — CurateCMGate
curate_magnitude.py      (44 lines) — CurateMagnitude
curate_patchwork.py      (80 lines) — CuratePatchwork, CurateGatedPatchwork
fuse.py                  (53 lines) — FuseGeometric
geometric_transformer.py (1444 lines) — 11 CLASSES ← the problem
mutate_flow.py           (32 lines) — MutateFlow
mutate_relay.py          (41 lines) — MutateRelay
observe_svd.py           (74 lines) — ObserveSVD, ObserveSVDTokens
```

### Classes in geometric_transformer.py to Extract
Each should become its own file following the existing naming convention:

| Class | Lines | New File | Stage Category |
|-------|-------|----------|----------------|
| CMValidatedGate | ~80 | `curate_cm_gate.py` | Curation |
| GeoResidualBank | ~40 | `distinguish_nce_bank.py` | Distinction |
| FiLMLayer | ~30 | `context_film.py` | Context |
| CayleyOrthogonal | ~60 | `align_cayley.py` | Alignment |
| QuaternionCompose | ~50 | `compose_quaternion.py` | Composition |
| ManifoldProjection | ~20 | `project_manifold.py` | Projection |
| PositionGeometricContext | ~80 | `context_position_geometric.py` | Context |
| GeometricAttention | ~40 | `attend_geometric.py` | Attention |
| ContentAttention | ~30 | `attend_content.py` | Attention |
| GeometricTransformerLayer | ~200 | `layer_geometric.py` | Layer |
| GeometricTransformer | ~600 | `transformer_geometric.py` | Model |

Also extract standalone functions:
- `pairwise_distances_squared` → `core/curate/cm.py` or keep in `curate_cm_gate.py`
- `cayley_menger_det` → same
- `anchor_neighborhood_cm` → same
- `_perturb_target`, `_project_tangent`, `_compute_centroids_and_assign` → already in `core/curate/patchwork.py`

### Steps
1. Read `geometric_transformer.py` top to bottom, identify exact line ranges per class
2. Create each new file with proper imports
3. Update `pipeline/components/__init__.py` to export from new locations
4. Add backward-compatible imports in a trimmed `geometric_transformer.py` that re-exports
5. Run `python -c "from geolip_core.pipeline.components.geometric_transformer import *"` to verify
6. Run the Redux dry run script to verify nothing broke

### Critical: Preserve These Fixes
- `CayleyOrthogonal.precompute()` MUST have `@torch.no_grad()` decorator
- `CayleyOrthogonal.forward()` fallback path MUST have `with torch.no_grad()`
- `CMValidatedGate.precompute()` has `with torch.no_grad():` — preserve
- `GeometricTransformer.precompute_cm_gates()` has `@torch.no_grad()` — preserve

---

## Task 2: Curriculum Observer Prototype

### Concept: Episodic Crystallization
Instead of training the constellation jointly with the model (which creates co-dependent
"muddy" structure), train in stages:

```
Stage 0: Train transformer (CE only, no geometry) for N epochs
         → Freeze model
         → Crystallize constellation from stable embedding space
         
Stage 1: Train with crystallized constellation (full observer losses) for N epochs
         → Freeze model
         → Re-crystallize: Procrustes-align new embedding space to old constellation
         → Inherit stable anchors, prune/reassign dead ones
         
Stage K: Constellation inherits from K-1, refined by K's embedding geometry
```

### Key Components to Build

#### 1. `CrystallizationEngine`
Given a frozen model and a data loader:
- Collect embeddings from all training data (or large sample)
- Compute class centroids on S^(d-1)
- Place anchors at centroids (100 classes → 100 anchors)
- Remaining anchors → midpoints between most confused class pairs
- Compute CV of resulting constellation
- Return crystallized ConstellationObserver

```python
class CrystallizationEngine:
    def __init__(self, manifold_dim, n_anchors, n_classes):
        ...
    
    @torch.no_grad()
    def collect_embeddings(self, model, loader, project_fn, max_samples=50000):
        """Run model, collect (embedding, label) pairs."""
        ...
    
    def compute_centroids(self, embeddings, labels):
        """Class centroids on S^(d-1)."""
        ...
    
    def place_confusion_anchors(self, centroids, n_extra):
        """Remaining anchors at confused class-pair midpoints."""
        ...
    
    def crystallize(self, model, loader, project_fn):
        """Full crystallization: collect → centroids → place → validate."""
        ...
        return constellation, report
```

#### 2. `StageAligner` (Procrustes between stages)
Given Stage K-1's constellation and Stage K's embedding space:
- Procrustes-align old anchors into new space
- Identify stable anchors (survived rotation with low residual)
- Identify drifting anchors (high residual → need updating)
- Identify dead anchors (no samples assigned → reassign)
- Return aligned constellation + structural report

Uses: `geolip_core.linalg.procrustes` and `geolip_core.core.align.procrustes`

```python
class StageAligner:
    def __init__(self, manifold_dim):
        ...
    
    @torch.no_grad()
    def align(self, old_constellation, new_embeddings, new_labels):
        """Procrustes-align old constellation to new embedding space."""
        # R, scale = procrustes(old_anchors, new_centroids)
        # aligned = old_anchors @ R.T
        # residuals per anchor
        # stable = residual < threshold
        # dead = assignment count < min_util
        ...
        return aligned_constellation, report
    
    def structural_report(self, old_constellation, new_constellation):
        """SVD of drift vectors, eigenspectrum comparison."""
        ...
```

#### 3. `CurriculumTrainer`
Orchestrates the stage loop:
```python
class CurriculumTrainer:
    def __init__(self, model, config):
        self.engine = CrystallizationEngine(...)
        self.aligner = StageAligner(...)
    
    def run_stage(self, stage_num, epochs, loader):
        """One stage: train → freeze → crystallize → align."""
        if stage_num == 0:
            # CE only, no constellation losses
            train(self.model, loader, epochs, losses=['ce'])
        else:
            # Full observer losses with crystallized constellation
            train(self.model, loader, epochs, losses='all')
        
        # Crystallize
        constellation, report = self.engine.crystallize(self.model, loader)
        
        if stage_num > 0:
            # Align to previous
            aligned, align_report = self.aligner.align(
                self.prev_constellation, embeddings, labels)
            constellation = aligned
            report.update(align_report)
        
        self.prev_constellation = constellation
        return report
    
    def train_curriculum(self, loader, stages):
        """Full curriculum: [(n_epochs, config), ...]"""
        for i, (epochs, cfg) in enumerate(stages):
            report = self.run_stage(i, epochs, loader)
            log(report)
```

### Where to Build
- `geolip_core/core/align/crystallize.py` — CrystallizationEngine
- `geolip_core/core/align/stage_align.py` — StageAligner  
- `geolip_core/training/curriculum.py` — CurriculumTrainer (new directory)

### Dependencies
- `geolip_core.linalg.procrustes` — Procrustes rotation
- `geolip_core.core.associate.constellation` — ConstellationObserver
- `geolip_core.core.curate.patchwork` — AnchorPush (for comparison)
- `geolip_core.core.distinguish.losses` — cv_loss, cv_metric, spread_loss
- `geolip_core.analysis.geometric` — structural analysis tools

### Validation
- Run Stage 0 (CE only) on CIFAR-100 for 50 epochs
- Crystallize constellation from embeddings
- Verify CV_anc is in 0.20-0.23 band (should be by construction)
- Run Stage 1 (full losses) for 50 epochs
- Verify accuracy improves over non-curriculum baseline (Redux at 61%)
- Verify constellation stability via Procrustes residual between stages

---

## Task 3: System Documentation

### Output: `SYSTEM.md` — comprehensive reference for all geolip-core utilities

### Structure

```markdown
# GeoLIP Core — System Reference

## Architecture Overview
- Six-stage paradigm diagram
- geofractal dependency (BaseRouter, BaseTower, TorchComponent, WideRouter)
- Design philosophy: observe, don't participate

## Package Map

### geolip_core/linalg/ — Compilable Linear Algebra
- eigh.py: FLEigh (Faddeev-LeVerrier hybrid eigendecomposition)
  - 84/84 mathematical purity, 1.73× cuSOLVER speed, 40× less memory
  - CUDA-graph-safe (no cuSOLVER calls)
  - Usage: LA.eigh(G, method='fl')
- svd.py: gram_fl_eigh_svd
  - SVD via Gram matrix + FL eigh, fully compilable
  - Usage: LA.svd(A, method='gram_eigh')
- procrustes.py: Orthogonal Procrustes alignment
- newton_schulz.py: Newton-Schulz matrix normalization
- _backend.py: Dispatch layer (fl vs cusolver)

### geolip_core/core/ — Pure Math Primitives

#### core/input/
- svd.py: SVDObserver (spatial), SVDTokenObserver (sequential)
  - extract_features(): S → (s_norm, vh_diag, vh_offdiag, entropy)
  - compute_novelty(): deviation from EMA
- scatter.py: Scattering transform features
- spectral.py: Spectral feature extraction

#### core/associate/
- constellation.py:
  - Constellation: learnable anchors on S^(d-1)
  - ConstellationAssociation: triangulate + assign
  - ConstellationCuration: patchwork + bridge
  - ConstellationObserver: compose Association + Curation
- relay.py: RelayLayer — relay-stack patchwork layers
- route.py: FlowAttention — 3-step Euler ODE flow in tangent space

#### core/curate/
- patchwork.py:
  - Patchwork: round-robin compartment MLP (stride slicing, CUDA-graph-safe)
  - MagnitudeFlow: relay-stack magnitude prediction
  - AnchorPush: non-gradient momentum anchor repositioning
- flows.py:
  - FlowQuaternion, FlowQuaternionLite, FlowVelocity,
    FlowMagnitude, FlowOrbital, FlowAlignment
  - FlowEnsemble: multi-opinion weighted fusion
  - FLOW_REGISTRY + build_flow()
- gate.py: gating utilities

#### core/distinguish/
- losses.py — ALL loss functions:
  - cv_loss(emb, target): pentachoron volume CV on embeddings
  - cv_metric(emb): non-differentiable CV for monitoring
  - spread_loss(anchors): anchor repulsion
  - nce_loss(z1, z2, temp): InfoNCE contrastive
  - ce_loss_paired(logits1, logits2, targets): paired cross-entropy
  - bridge_loss_paired(): assignment prediction
  - assign_bce_loss(): crispness (BCE toward hard nearest)
  - assign_nce_loss(): assignment consistency between views
  - attraction_loss(): pull embeddings toward nearest anchor
  - observer_loss(): full observer self-organization recipe
  - three_domain_loss(): external + geometric + internal

#### core/align/
- procrustes.py: Orthogonal Procrustes alignment utilities

#### core/util.py
- make_activation(): activation factory
- param_count(), model_summary(): diagnostic utilities

### geolip_core/pipeline/ — TorchComponent Wrappers

#### pipeline/observer.py — Base Classes
- Input, Mutation, Association, Curation, Distinction (all TorchComponent)
- GeoLIP (BaseTower) — top-level tower

#### pipeline/components/ — Named Components
- (see component reorganization task for full list)
- Key: these are TorchComponent adapters around core/ primitives
- They add cache_set/cache_get, parent binding, on_attach lifecycle

#### pipeline/layer.py — Layer composition utilities
#### pipeline/backbone.py — Backbone adapters
#### pipeline/esm2_geometric.py — ESM-2 protein geometric pipeline

### geolip_core/example/ — Complete Working Examples
- constellation_encoder.py: ConstellationEncoder + GeoLIPEncoder
  - The original working architecture with MagnitudeFlow + AnchorPush
  - Reference implementation for new architectures
- conv_encoder.py, conv_svd_encoder.py, conv_scatter_encoder.py
- spectral_encoder.py, transformer_svd_encoder.py

### geolip_core/analysis/
- geometric.py: structural analysis tools

### geolip_core/utils/
- kernel.py: GPU kernel utilities
- memory.py: memory tracking
- triton/: Triton kernel implementations (FL eigh)

## Key Patterns

### Observer Pattern
observe() → triangulate against anchors → curate through patchwork → distinguish
The observer doesn't modify the data — it measures geometric relationships.

### Precompute/Invalidate Cycle
```python
model.invalidate()    # mark caches stale
model.precompute()    # recompute expensive ops (LA.det, LA.solve)
# ... N forward/backward steps ...
model.invalidate()    # ready for next cycle
```

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

## Empirical Results
- Redux v8: 61% CIFAR-100, 6.8M params, CV_anc=0.212 (in band)
- ConstellationEncoder: stable CV convergence with relay + push
- Procrustes survey: 17 models, QK 0.500 lock universal, VAE 70-76% alignable
- GEOLIP-Bertenstein: perfect retrieval on 40K+ pairs, 1 epoch, 1 layer
- FL Eigh: 84/84 purity, 1.73× cuSOLVER, 40× less memory
```

---

## Execution Order for Claude Code

1. **Read CLAUDE.md** first (critical rules)
2. **Task 1**: Component reorg — mechanical, low risk, improves all downstream work
3. **Task 3**: System docs — generates while reading code for Task 1
4. **Task 2**: Curriculum observer — the creative work, needs clean codebase from Task 1

For Task 2, start with `CrystallizationEngine` alone and validate it produces
a constellation in the CV band from CIFAR-100 embeddings. Then add `StageAligner`.
`CurriculumTrainer` comes last and ties everything together.