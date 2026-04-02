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

Triton is optional — fused SVD kernels activate automatically if installed. CuPy is optional — CUDA eigendecomposition kernels activate if installed. Everything falls back to PyTorch without either.

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
├── linalg/                        Geometric linear algebra primitives
│   ├── __init__.py                    Drop-in torch.linalg replacement
│   ├── _backend.py                    CUDA/Triton/CuPy detection, one-time warning
│   ├── eigh.py                        FL Hybrid Eigendecomposition (FLEigh)
│   ├── svd.py                         Batched thin SVD with FL eigh integration
│   ├── newton_schulz.py               Iterative inverse square root (pure bmm)
│   └── procrustes.py                  Subspace-preserving Procrustes alignment
│
├── example/                       Working models built with the pipeline
├── analysis/                      Diagnostic tools
└── utils/                         Engineering infrastructure
    ├── kernel.py                      Triton SVD N=2,3, gram_eigh, Procrustes math
    ├── cuda/                          CuPy/NVRTC eigendecomposition kernels
    │   └── fl_eigh_cuda.py                Per-N generated CUDA kernels (n=3-16)
    ├── triton/                        Generated Triton kernels (experimental)
    │   ├── fl_eigh_gen.py                 Kernel source generator
    │   └── fl_eigh_n6.py                 Pre-generated n=6 kernel (7465 lines)
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

### Geometric Linear Algebra

`geolip.linalg` is a drop-in replacement for `torch.linalg`. Our implementations override where we have something better. Everything else transparently proxies to PyTorch.

```python
import geolip.linalg as LA

# Our implementations (auto-dispatch to best available)
vals, vecs = LA.eigh(A)           # FL pipeline n≤12, cuSOLVER n>12
U, S, Vh = LA.svd(A)              # Triton n=2,3 → FL n≤12 → cuSOLVER

# These pass through to torch.linalg (zero overhead)
x = LA.solve(A, b)
L = LA.cholesky(A)
n = LA.norm(x)

# Configuration
LA.backend.status()               # print available features
LA.backend.use_fl_eigh = False    # disable FL, use cuSOLVER everywhere
```

**FL Hybrid Eigendecomposition** — Faddeev-LeVerrier characteristic polynomial → Laguerre root-finding → Newton-Schulz orthogonalization → Rayleigh quotient refinement. Wins 84/84 mathematical purity metrics against cuSOLVER across n=3-12. Zero graph breaks under `torch.compile(fullgraph=True)`. 40× less memory.

```python
from geolip.linalg import FLEigh

# Compiled training loop — zero graph breaks
solver = torch.compile(FLEigh(), fullgraph=True)
eigenvalues, eigenvectors = solver(cm_matrices)  # [B, 6, 6] → [B, 6], [B, 6, 6]
```

**CUDA Eigendecomposition Kernel** — Per-matrix CUDA kernel via CuPy/NVRTC. One thread per matrix, entire pipeline in registers. Flat batch scaling — 1.73× cuSOLVER at B=16,384, runs where cuSOLVER OOMs at B=32,768.

```python
from geolip.utils.cuda.fl_eigh_cuda import fl_eigh_cuda

# High-batch: 0.7MB vs cuSOLVER's 1,099MB
eigenvalues, eigenvectors = fl_eigh_cuda(A)  # any n in 3-16
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

### Geometric Linear Algebra (`linalg/`)

Drop-in `torch.linalg` replacement with geometric optimizations. Anything not overridden proxies transparently to PyTorch.

| Function | Implementation | Performance |
|---|---|---|
| `eigh(A)` | FL Hybrid (n≤12), cuSOLVER fallback | 84/84 purity, 40× less memory, zero graph breaks |
| `svd(A)` | Triton N=2,3 → FL eigh N≤12 → cuSOLVER | 3,850× at N=2, compilable |
| `newton_schulz_invsqrt(G)` | Pure bmm iteration | Zero eigensolvers, quadratic convergence |
| `procrustes(src, tgt)` | Subspace-preserving rotation | 1.000 NN agreement at N=32-128 |
| Everything else | `torch.linalg` passthrough | Zero overhead |

Backend auto-detection with single warning on first fallback:

```python
from geolip.linalg import backend
backend.status()
# geolip.linalg backend:
#   CUDA:       yes
#   Triton:     3.6.0
#   FL eigh:    enabled
#   Triton SVD: enabled
#   GPU:        NVIDIA RTX PRO 6000 Blackwell Server Edition
```

| Dispatch condition | Implementation | Why |
|---|---|---|
| n≤4, CuPy available, B≥512 | CUDA kernel | 2-7× cuSOLVER |
| n≤6, CuPy available, B≥8,192 | CUDA kernel | 1.7× cuSOLVER, flat scaling |
| n≤12, CUDA | FL Precise (compiled) | 84/84 purity, 40× less memory |
| n>12 | `torch.linalg.eigh` | FL conditioning degrades |

### Key Core Modules

**SVD Kernel** (`utils/kernel.py`) — Fused Triton kernels for batched thin SVD:

| N | Time | vs torch |
|---|---|---|
| 2 | 0.021ms | 3,850× |
| 3 | 0.022ms | 5,488× |
| 8 | 0.290ms | 584× |
| 32 | 0.781ms | 388× |

**FL Hybrid Eigh** (`linalg/eigh.py`) — Five-phase compilable eigendecomposition:

| Phase | Method | Precision |
|---|---|---|
| 1. Characteristic polynomial | Faddeev-LeVerrier recurrence | fp64 |
| 2. Root-finding | Laguerre + synthetic deflation | fp32/fp64 adaptive |
| 3. Eigenvectors | FL adjugate Horner evaluation | fp64, chunked for n>6 |
| 4. Orthogonalization | Newton-Schulz polar iteration | fp32, 2 iterations |
| 5. Eigenvalue refinement | Rayleigh quotient | fp32, 2 bmm |

Benchmarked on NVIDIA RTX PRO 6000 Blackwell (B=4,096, n=6):

| Method | Time | vs cuSOLVER | Memory |
|---|---|---|---|
| cuSOLVER | 241 µs | 1.00× | 1,099 MB |
| FL Precise compiled | 350 µs | 0.69× | 32 MB |
| FL Precise + CUDA Graph | 287 µs | 0.84× | 32 MB |
| CUDA kernel (n=6, B=16K) | 429 µs | 1.73× | 0.7 MB |
| CUDA kernel (n=3, B=4K) | 45 µs | 3.06× | 0.7 MB |

**CUDA Eigh Kernel** (`utils/cuda/fl_eigh_cuda.py`) — Generated per matrix size via Python template, compiled by NVRTC at runtime, cached at `~/.cupy/kernel_cache/`. One thread per matrix, zero intermediate memory, flat batch scaling.

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

Optional:
- `triton >= 2.1` — fused SVD kernels for N=2,3
- `cupy-cuda12x` — CUDA eigendecomposition kernels via NVRTC
- `kymatio` — scattering transforms

## Ecosystem

- [geofractal](https://github.com/AbstractEyes/geofractal) — Router/tower/component composition framework
- [wide-compiler](https://github.com/AbstractEyes/wide-compiler) — N-model batched fusion via grouped operations
- [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) — Full GeoLIP package
- [FL Eigh Article](https://huggingface.co/blog/AbstractPhil/fl-hybrid-eigendecomposition) — Compilable eigendecomposition beating cuSOLVER
- [SVD Kernel Article](https://huggingface.co/blog/AbstractPhil/svd-triton-kernel-optimization) — Engineering specification
- [SVD Experiment Journey](https://huggingface.co/blog/AbstractPhil/svd-experiment-journey) — Development map
- [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) — Multi-expert geometric fusion
- [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) — Cross-model alignment study

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil). Apache 2.0.*