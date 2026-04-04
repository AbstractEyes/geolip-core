# GeoLIP Core — Claude Code Project Context

This is a temporary CLAUDE.md file to provide minimal context for the codebase.

Check the CLAUDE_TASKS.md for the current task list and project roadmap meant
for Claude Code to modify.

## Owner
Phil (AbstractPhil / AbstractEyes / HuggingFace: AbstractPhil / GitHub: AbstractEyes)
Independent AI researcher. Synesthesia, ADHD. Pacifist. Builds openly.

## What This Is
geolip-core is a geometric deep learning framework grounded in unit hypersphere geometry.
Core primitive: constellations of learnable anchors on S^(d-1), triangulated via
Cayley-Menger determinants, curated through patchwork compartments, and distinguished
through observer losses. The framework observes, measures, and navigates high-dimensional
manifolds — it is a geometric measurement instrument, not just a classifier.

## Repositories
- `geolip-core` (this repo): pip-installable geometric observation framework
- `geofractal`: Router→Tower→Component architecture (dependency)
- `glip-autoencoder`: Original GLIP research
- `procrustes-analysis`: 17-model structural analysis

## Validated Constants — DO NOT CHANGE
- **CV pentachoron band**: 0.20–0.23 (universal attractor, all architectures/modalities)
- **Binding constant**: 0.29154 (complement 0.70846) — structural/physical phase boundary
- These are empirically validated theory-level, not hypotheses

## Architecture: Six-Stage Observer Paradigm
```
Input → Mutation → Association → Curation → Distinction → Loss
  │         │           │            │            │          │
  SVD    FlowAttn   Constellation  Patchwork   ClassHead  observer_loss
  │         │        + CM Gate       + Bridge              + cv_loss
  │         │           │            │                     + spread_loss
  └─────────┴───────────┴────────────┘
        All observe a SHARED constellation
```

## CRITICAL RULES — READ EVERY SESSION

### torch.compile
- **NEVER return dicts with multiple tensor values from compiled functions.** Each dict key
  is a separate guarded output — N keys = N× overhead EVERY call. Return `torch.stack([...])`
  and unpack outside. This caused 10× slowdown (3s→30s/epoch).
- Applies to both `default` and `reduce-overhead` modes.

### Precompute / Buffer Patterns
- **Any `precompute()` that writes to buffers via `.copy_()` from learnable parameters MUST
  use `@torch.no_grad()`.** Without it, `.copy_()` links the buffer to the parameter's
  autograd graph — next batch's forward inherits stale graph, backward crashes with
  "second time through freed graph." Root cause of multi-session CayleyOrthogonal debug.
- Also add `@torch.no_grad()` to fallback paths in `forward()` if they call `.copy_()`.

### Gradient Clipping
- Use `min(max(task_loss_value, 1.0), 2.0)` instead of hard 1.0 or unbounded. Multi-loss
  training (CE + NCE + bridge + assign etc.) at hard clip=1.0 starves classification.
  Unbounded at CE=4.6 causes NaN.

### Constellation Architecture
- `forward_paired` extracts from `geo_states[-1]` ONLY. Earlier layers' anchors get zero
  direct loss signal. With N layers, effectively only 1 is geometrically supervised.
- Fix: shared constellation across all layers (Redux architecture), OR per-layer losses.

### Precompute Frequency
- CM gate (`LA.det` on 128 simplices) and Cayley rotation (`LA.solve`) don't need every
  batch. Anchors move ~0.01% per step. Every 10-20 batches is fine.

### Spatial Structure
- NEVER use global average pooling in geometric encoders — destroys spatial structure,
  drops accuracy from ~70% to ~29%.

### Anchor Push
- Momentum push (AnchorPush) with alpha=0.05, beta=0.02 is stable for CIFAR-100.
- alpha=0.1, beta=0.05 causes NaN at ~E75 due to excessive drift.
- Buffer size 8192 → ~6 pushes/epoch at B=128 on CIFAR-100.

## Current State (April 2026)

### Working: GeoTransformer Redux v8
- File: `geometric_transformer_redux.py` (standalone training script)
- 61% CIFAR-100, 6.8M params, 4 layers, shared constellation
- CV_anc converged to 0.212 (in band), 126/128 anchors active
- 19s/epoch on Blackwell RTX PRO 6000

### Known Issues
- `pipeline/components/geometric_transformer.py` is a 1444-line monolith with 11 classes
  that need splitting into separate files
- Several components are raw `nn.Module` instead of `TorchComponent` (CMValidatedGate,
  GeoResidualBank, ConstellationObserver) — inconsistent with the geofractal paradigm
- The constellation gets "muddy" — retains alignment from earlier less-accurate states
  instead of refining. Co-dependent with the transformer during training.

### Next: Curriculum Observer (Episodic Crystallization)
- Separate the constellation from gradient-based training
- Train transformer (CE only) → freeze → crystallize constellation from stable embeddings
- Procrustes-align between stages, inherit stable anchors, prune dead ones
- The constellation becomes a measurement instrument, not a training participant

## File Locations
- Core math: `geolip_core/core/` (associate, curate, distinguish, input, align)
- Pipeline wrappers: `geolip_core/pipeline/` (TorchComponent adapters)
- Linear algebra: `geolip_core/linalg/` (FL eigh, SVD, Procrustes, Newton-Schulz)
- Examples: `geolip_core/example/` (ConstellationEncoder, ConvEncoder, etc.)
- Analysis: `geolip_core/analysis/geometric.py`