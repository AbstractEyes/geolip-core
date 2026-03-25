# geolip-core

A geometric support system for deep learning models.

Models train blind. They receive a loss signal that says "you were wrong" and a gradient that says "go this direction." They have no structural self-awareness — they can't see that their representations are collapsing, that their features are redundant, or that half their capacity is dead.

This package gives them that. Not by adding losses that punish problems after they happen, but by architecturally making those problems observable, measurable, and correctable in real time.

Part of the [GeoLIP](https://github.com/AbstractEyes/glip-autoencoder) ecosystem.

## Install

```bash
pip install "git+https://github.com/AbstractEyes/geolip-core.git"
```

Triton is optional — fused SVD kernels activate automatically if installed. Everything falls back to PyTorch without it.

## What This Does

The geometric substrate attaches to any backbone (conv, transformer, hybrid) and provides:

- **Structural observation** — SVD decomposition of feature maps reveals energy distribution, rotation patterns, and novelty. Detached from the backward pass, zero interference with training.
- **Geometric association** — Constellation anchors on S^(d-1) triangulate embeddings against a stable reference frame. The anchors define the coordinate system.
- **Quality-aware curation** — Cayley-Menger validity gates measure whether each anchor forms a well-conditioned simplex with the input. Simplex volume IS the attention score. Invalid geometry gets suppressed before it reaches the interpreter.
- **Targeted modulation** — Patchwork compartments interpret curated associations. Channel modulation adjusts backbone features based on geometric health. Local intervention, not global perturbation.

Empirically validated: +3 accuracy points on CIFAR-100 across both convolutional and transformer backbones, from 150K additional parameters.

## Package Structure

```
geolip_core/
├── core/               Geometric behaviors (the five stages)
│   ├── input/              Data-type ingestion and observation
│   ├── associate/          Measure relationships to reference frame
│   ├── curate/             Select what matters from associations
│   ├── align/              How spaces relate to each other
│   └── distinguish/        Task-specific output and training signal
│
├── pipeline/           Composed geometric substrates
│   ├── observer.py         Stage interfaces (Input, Association, Curation, ...)
│   ├── layer.py            ConstellationLayer — one depth of observation
│   └── backbone.py         GeometricBackbone — multi-depth stack
│
├── example/            Working models built with the pipeline
├── analysis/           Diagnostic tools
└── utils/              Engineering infrastructure (Triton kernels, linalg)
```

### The Five Stages

Every component lives in the directory matching its primary purpose.

| Stage | Directory | Purpose |
|---|---|---|
| **Input** | `core/input/` | Ingest and decompose external signals into geometric primitives |
| **Associate** | `core/associate/` | Measure relationships to a reference frame |
| **Curate** | `core/curate/` | Select what matters from those measurements |
| **Align** | `core/align/` | Relate two geometric spaces to each other |
| **Distinguish** | `core/distinguish/` | Task-specific output |

A component may perform hundreds of internal steps. It is classified by what it exists to accomplish, not by what it computes along the way.

### Design Principle: Architecture Before Loss

The geometric substrate is structural, not supervisory. When a problem arises:

1. First ask: can the architecture prevent this?
2. If yes: structural fix — gate, projection, initialization, detach boundary.
3. If no: then introduce a loss, minimal and targeted.

The CM gate doesn't need a "simplex validity loss" to penalize degenerate anchors. It architecturally suppresses them. Subspace-preserving Procrustes doesn't need an "alignment loss." The rotation is mathematically exact by construction. Repulsion-initialized anchors don't need a spread loss if the gate prevents collapse.

## Quick Start

```python
# Individual behaviors
from geolip_core.core import Constellation, Patchwork, AnchorGate, SVDObserver

# Composed pipeline
from geolip_core.pipeline import ConstellationLayer, GeometricBackbone

# Engineering utilities
from geolip_core.utils import gram_eigh_svd, batched_procrustes
```

### Attach to a Backbone

```python
from geolip_core.pipeline import GeometricBackbone

# Define depth specs: (channels, spatial_size) at each stage
geo = GeometricBackbone(
    stages=[(64, 32), (128, 16), (256, 8), (384, 4)],
    n_anchors=32, svd_rank=24, gate_strategy='cm_gate',
)

# In your forward pass:
features = [stage(h) for stage, h in zip(conv_stages, intermediates)]
modulated, geo_state, observations = geo(features)
# modulated features replace originals; geo_state feeds your classifier
```

### Use Individual Components

```python
from geolip_core.core import SVDObserver, AnchorGate, Patchwork

# Observe structure
svd = SVDObserver(in_channels=384, svd_rank=24)
S, Vh, features, novelty = svd(conv_features)

# Gate by geometric validity
gate = AnchorGate(n_anchors=32, dim=256, strategy='cm_gate')
gate_values, assignment, info = gate(embedding, anchors, triangulation)

# Interpret curated associations
pw = Patchwork(n_anchors=32, n_comp=8, d_comp=64)
interpreted = pw(triangulation * gate_values)
```

### Align Two Spaces

```python
from geolip_core.core import ProcrustesAlignment

aligner = ProcrustesAlignment(dim=384, rank=24)
aligned, info = aligner(source_embeddings, target_embeddings)
# info['method'] = 'subspace' for dim > 32, 'full' otherwise
# 1.000 nearest-neighbor agreement with full Procrustes
```

## Key Components

### SVD Kernel (`utils/kernel.py`)

Fused Triton kernels for batched thin SVD. 5,000× faster than `torch.linalg.svd` for small N.

| N | Time | vs torch |
|---|---|---|
| 2 | 0.021ms | 3,850× |
| 3 | 0.022ms | 5,488× |
| 8 | 0.290ms | 584× |
| 32 | 0.781ms | 388× |

Auto-dispatches: Triton for N≤3, Gram+eigh for N=4-32. AMP-safe (disables autocast around linalg). See the [engineering article](https://huggingface.co/blog/AbstractPhil/svd-triton-kernel-optimization) for the full specification.

### CM Validity Gate (`core/curate/gate.py`)

Cayley-Menger determinant as geometric attention. For each anchor, forms a simplex with the embedding and its neighbors. The simplex volume is the relevance score — fat simplex means the anchor provides genuine geometric information. Sliver simplex means it's redundant.

Strategies: `round_robin` (baseline), `cm_gate` (soft sigmoid), `top_k` (hard selection), `top_p` (nucleus).

### Subspace Procrustes (`core/align/procrustes.py`)

For N > 32, projects to rank-24, aligns in the projected space, lifts back preserving the orthogonal complement exactly. Validated: 1.000 nearest-neighbor agreement with full Procrustes across all tested configurations (N=32-128, k=8-64). Three matmuls, sub-millisecond.

### Constellation (`core/associate/constellation.py`)

Learned anchors on S^(d-1). The primary state — the reference frame that everything else measures against. Repulsion-initialized for maximal coverage. Detached from task gradients; positioned by geometric structure only.

## Empirical Constants

| Constant | Value | Observed across |
|---|---|---|
| CV pentachoron band | 0.20–0.23 | 17+ architectures, all modalities |
| Binding/separation boundary | 0.29154 / 0.70846 | MinimalShunts, CLIP, T5, alpha convergence |
| Effective geometric dimension | 16 (S^15) | Validated in patchwork and anchor experiments |
| Irreducible CV minimum | 0.125 | Theoretical lower bound on sphere |

## Requirements

```
torch >= 2.0
```

Optional: `triton >= 2.1` (fused SVD kernels), `geofractal` (tower composition), `kymatio` (scattering).

## Ecosystem

- [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) — Full GeoLIP package (PyPI: `geolip`)
- [SVD Kernel Article](https://huggingface.co/blog/AbstractPhil/svd-triton-kernel-optimization) — Engineering specification
- [SVD Experiment Journey](https://huggingface.co/blog/AbstractPhil/svd-experiment-journey) — Development map with every wrong turn documented
- [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) — Multi-expert geometric fusion
- [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) — Cross-model alignment study

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil). Apache 2.0.*