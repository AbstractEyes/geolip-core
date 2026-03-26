# geolip-core

A geometric support system for deep learning models.

Models train blind. They get a loss signal and a gradient. They have no structural self-awareness — they can't see their representations collapsing, their features going redundant, or their capacity dying.

This package gives them that. Not by adding losses that punish problems after they happen, but by architecturally making those problems observable, measurable, and correctable in real time. A model hospital — diagnostics, monitoring, intervention, and support for models that can't diagnose themselves.

Built on the [geofractal](https://github.com/AbstractEyes/geofractal) router system. Part of the [GeoLIP](https://github.com/AbstractEyes/glip-autoencoder) ecosystem.

## Install

```bash
pip install "git+https://github.com/AbstractEyes/geofractal.git"
pip install "git+https://github.com/AbstractEyes/geolip-core.git"
```

Triton is optional — fused SVD kernels activate automatically if installed. Everything falls back to PyTorch without it.

## Architecture

Three layers, zero drift between them:

```
core/                         THE GEOMETRY (nn.Module)
  Standalone modules. No framework dependency beyond PyTorch.
  These do the math.

pipeline/components/          THE CACHE ADAPTERS (TorchComponent)
  Thin wrappers around core modules. Each owns self.inner,
  reads from parent cache, delegates to inner, writes results.
  These wire the bus.

pipeline/                     THE COMPOSITIONS (BaseTower)
  Orchestrate component execution order via forward().
  These define the flow.
```

A model is what you build FROM a pipeline. The pipeline composes behaviors. The behaviors live in core.

## Package Structure

```
geolip_core/
├── core/                          Geometric behaviors (five stages)
│   ├── input/                         Data-type ingestion and observation
│   │   └── svd.py                         SVDObserver, SVDTokenObserver
│   ├── associate/                     Measure relationships to reference frame
│   │   ├── constellation.py               Constellation, ConstellationAssociation
│   │   ├── relay.py                       ConstellationRelay (O(S) mutation)
│   │   └── route.py                       FlowAttention (ODE mutation)
│   ├── curate/                        Select what matters
│   │   ├── gate.py                        AnchorGate, GatedPatchwork (CM validity)
│   │   └── patchwork.py                   Patchwork, MagnitudeFlow
│   ├── align/                         How spaces relate
│   │   └── procrustes.py                  ProcrustesAlignment (subspace-preserving)
│   ├── distinguish/                   Task-specific output
│   │   └── losses.py                      CE, CV, spread, observer_loss
│   └── util.py                        Activations, autograd, constants
│
├── pipeline/                      Composed geometric substrates
│   ├── components/                    TorchComponent wrappers (cache adapters)
│   │   ├── observe_svd.py                 ObserveSVD, ObserveSVDTokens
│   │   ├── associate_constellation.py     AssociateConstellation
│   │   ├── mutate_relay.py                MutateRelay
│   │   ├── mutate_flow.py                 MutateFlow
│   │   ├── curate_gate.py                 CurateCMGate
│   │   ├── curate_patchwork.py            CuratePatchwork, CurateGatedPatchwork
│   │   ├── curate_magnitude.py            CurateMagnitude
│   │   ├── align_procrustes.py            AlignProcrustes
│   │   └── fuse.py                        FuseGeometric
│   ├── observer.py                    Stage interfaces (Input, Association, ...)
│   ├── arbitrary_feature.py           GeometricPipeline (token → geo feature)
│   ├── layer.py                       ConstellationLayer (one depth)
│   └── backbone.py                    GeometricBackbone (multi-depth)
│
├── example/                       Working models built with the pipeline
├── analysis/                      Diagnostic tools
└── utils/                         Engineering infrastructure
    ├── kernel.py                      Triton SVD, gram_eigh, Procrustes math
    └── memory.py                      EmbeddingBuffer
```

### The Five Stages

Every component in `core/` lives in the directory matching its primary purpose. A component may perform hundreds of internal steps. It is classified by what it exists to accomplish, not by what it computes along the way.

| Stage | Directory | Purpose |
|---|---|---|
| **Input** | `core/input/` | Ingest and decompose external signals into geometric primitives |
| **Associate** | `core/associate/` | Measure relationships to a reference frame |
| **Curate** | `core/curate/` | Select what matters from those measurements |
| **Align** | `core/align/` | Relate two geometric spaces to each other |
| **Distinguish** | `core/distinguish/` | Task-specific output |

### Design Principle: Architecture Before Loss

When a problem arises:

1. First ask: can the architecture prevent this?
2. If yes: structural fix — gate, projection, initialization, detach boundary.
3. If no: then introduce a loss, minimal and targeted.

The CM gate doesn't need a validity loss. It architecturally suppresses degenerate anchors. Subspace Procrustes doesn't need an alignment loss. The rotation is exact by construction.

## Quick Start

### Composable Pipeline (geofractal router)

```python
from geolip_core.pipeline.arbitrary_feature import GeometricPipeline
from geolip_core.pipeline.components import CurateCMGate

# Build pipeline: (B, 5, 512) → (B, feature_dim)
pipe = GeometricPipeline('geo', seq_len=5, input_dim=512, n_anchors=32)
features = pipe(x)

# Inspect intermediates via cache
diag = pipe.get_diagnostics()
print(diag['gate_info'])    # CM validity stats
print(diag['svd_S'])        # singular values
pipe.cache_clear()          # managed lifecycle

# Swap a stage at runtime
pipe.detach('curate_gate')
pipe.attach('curate_gate', CurateCMGate('curate_gate', 32, 256, strategy='top_k'))
```

### Individual Components

```python
from geolip_core.pipeline.components import (
    ObserveSVDTokens, AssociateConstellation,
    CurateCMGate, CuratePatchwork, AlignProcrustes,
)

# Each wraps a core module + provides cache wiring
observe = ObserveSVDTokens('obs', seq_len=5)
assoc = AssociateConstellation('assoc', dim=256, n_anchors=32)
gate = CurateCMGate('gate', n_anchors=32, embed_dim=256, strategy='cm_gate')
```

### Core Modules Directly

```python
from geolip_core.core import SVDObserver, SVDTokenObserver, Constellation, AnchorGate, Patchwork

# No router, no cache — just PyTorch modules
svd = SVDObserver(in_channels=384, svd_rank=24)
S, Vh, features, novelty = svd(conv_features)

gate = AnchorGate(n_anchors=32, dim=256, strategy='cm_gate')
gate_values, assignment, info = gate(embedding, anchors, triangulation)
```

### Engineering Utilities

```python
from geolip_core.utils import gram_eigh_svd, batched_procrustes

# 5000× faster than torch.linalg.svd for small N
U, S, Vh = gram_eigh_svd(features)

# Subspace-preserving alignment
aligned, info = batched_procrustes(source, target, rank=24)
```

## Component Catalog

### Pipeline Components (TorchComponent wrappers)

Each reads from and writes to the parent router's cache. Core modules do the math.

| Component | Wraps | Stage | Cache writes |
|---|---|---|---|
| `ObserveSVD` | `SVDObserver` | Input | `svd_S`, `svd_Vh`, `svd_features`, `svd_novelty` |
| `ObserveSVDTokens` | `SVDTokenObserver` | Input | same |
| `AssociateConstellation` | `ConstellationAssociation` | Associate | `embedding`, `anchors_n`, `cos`, `tri`, `nearest`, `assignment` |
| `MutateRelay` | `ConstellationRelay` | Mutation | `relay_output`, `relay_tri` |
| `MutateFlow` | `FlowAttention` | Mutation | `flow_output` |
| `CurateCMGate` | `AnchorGate` | Curate | `gate_values`, `gate_info`, `tri_gated` |
| `CuratePatchwork` | `Patchwork` | Curate | `patchwork` |
| `CurateGatedPatchwork` | `GatedPatchwork` | Curate | `patchwork`, `gate_info` |
| `CurateMagnitude` | `MagnitudeFlow` | Curate | `mag_anchors`, `mag_comp` |
| `AlignProcrustes` | `ProcrustesAlignment` | Align | `aligned`, `alignment_info` |
| `FuseGeometric` | (pipeline-specific) | Fuse | `svd_context`, `geo_features` |

### Key Core Modules

**SVD Kernel** (`utils/kernel.py`) — Fused Triton kernels for batched thin SVD:

| N | Time | vs torch |
|---|---|---|
| 2 | 0.021ms | 3,850× |
| 3 | 0.022ms | 5,488× |
| 8 | 0.290ms | 584× |
| 32 | 0.781ms | 388× |

**CM Validity Gate** (`core/curate/gate.py`) — Cayley-Menger determinant as geometric attention. Simplex volume is the relevance score. Strategies: `round_robin`, `cm_gate`, `top_k`, `top_p`.

**Subspace Procrustes** (`core/align/procrustes.py`) — For N > 32, projects to rank-24, aligns, lifts back preserving orthogonal complement exactly. 1.000 NN agreement.

**Constellation** (`core/associate/constellation.py`) — Learned anchors on S^(d-1). The primary state. Repulsion-initialized, detached from task gradients.

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
geofractal @ git+https://github.com/AbstractEyes/geofractal.git
```

Optional: `triton >= 2.1` (fused SVD kernels), `kymatio` (scattering).

## Ecosystem

- [geofractal](https://github.com/AbstractEyes/geofractal) — Router/tower/component composition framework
- [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) — Full GeoLIP package
- [SVD Kernel Article](https://huggingface.co/blog/AbstractPhil/svd-triton-kernel-optimization) — Engineering specification
- [SVD Experiment Journey](https://huggingface.co/blog/AbstractPhil/svd-experiment-journey) — Development map
- [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) — Multi-expert geometric fusion
- [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) — Cross-model alignment study

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil). Apache 2.0.*