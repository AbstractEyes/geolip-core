# geolip-core

A geometric support system for deep learning models.

Models train blind. They get a loss signal and a gradient. They have no structural self-awareness -- they can't see their representations collapsing, their features going redundant, or their capacity dying.

This package gives them that. Not by adding losses that punish problems after they happen, but by architecturally making those problems observable, measurable, and correctable in real time. A model hospital -- diagnostics, monitoring, intervention, and support for models that can't diagnose themselves.

Built on the [geofractal](https://github.com/AbstractEyes/geofractal) router system. Part of the [GeoLIP](https://github.com/AbstractEyes/glip-autoencoder) ecosystem.

## Install

```bash
pip install "git+https://github.com/AbstractEyes/geofractal.git"
pip install "git+https://github.com/AbstractEyes/geolip-core.git"
```

Triton is optional -- fused SVD kernels activate automatically if installed. CuPy is optional -- CUDA eigendecomposition kernels activate if installed. Everything falls back to PyTorch without either.

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

### The Six-Stage Observer Paradigm

```
Input -> Mutation -> Association -> Curation -> Distinction -> Loss
  |         |           |            |            |          |
  SVD    FlowAttn   Constellation  Patchwork   ClassHead  observer_loss
  |         |        + CM Gate       + Bridge              + cv_loss
  |         |           |            |                     + spread_loss
  +---------+-----------+------------+
        All observe a SHARED constellation
```

| Stage | Directory | Purpose |
|---|---|---|
| **Input** | `core/input/` | Ingest and decompose external signals into geometric primitives |
| **Mutation** | `core/associate/` | Transform position on manifold (FlowAttention, Relay) |
| **Association** | `core/associate/` | Measure relationships to a reference frame (Constellation) |
| **Curation** | `core/curate/` | Select what matters from those measurements (Patchwork, CM Gate, Flows) |
| **Alignment** | `core/align/` | Relate two geometric spaces (Procrustes, Cayley, Crystallization) |
| **Distinction** | `core/distinguish/` | Task-specific output and loss functions |

### Design Principle: Architecture Before Loss

When a problem arises:

1. First ask: can the architecture prevent this?
2. If yes: structural fix -- gate, projection, initialization, detach boundary.
3. If no: then introduce a loss, minimal and targeted.

The CM gate doesn't need a validity loss. It architecturally suppresses degenerate anchors. Subspace Procrustes doesn't need an alignment loss. The rotation is exact by construction.

## Package Structure

```
geolip_core/
├── core/                          Geometric behaviors (six stages)
│   ├── input/                         Data-type ingestion and observation
│   │   ├── svd.py                         SVDObserver, SVDTokenObserver
│   │   ├── scatter.py                     Scattering transform features
│   │   └── spectral.py                    Spectral feature extraction
│   ├── associate/                     Measure relationships to reference frame
│   │   ├── constellation.py               Constellation, ConstellationAssociation,
│   │   │                                  ConstellationCuration, ConstellationObserver
│   │   ├── relay.py                       ConstellationRelay (O(S) mutation)
│   │   └── route.py                       FlowAttention (ODE mutation)
│   ├── curate/                        Select what matters
│   │   ├── gate.py                        AnchorGate, GatedPatchwork (CM validity)
│   │   ├── patchwork.py                   Patchwork, MagnitudeFlow, AnchorPush
│   │   └── flows.py                       FlowEnsemble (6 flow types), FLOW_REGISTRY
│   ├── align/                         How spaces relate
│   │   ├── procrustes.py                  ProcrustesAlignment (subspace-preserving)
│   │   ├── crystallize.py                 CrystallizationEngine (constellation from embeddings)
│   │   └── stage_align.py                 StageAligner (Procrustes between training stages)
│   ├── distinguish/                   Task-specific output
│   │   └── losses.py                      CE, CV, NCE, spread, bridge, observer_loss
│   └── util.py                        Activations, autograd, validated constants
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
│   │   ├── curate_cm_validated.py         CMValidatedGate (transformer-scale CM gate)
│   │   ├── align_procrustes.py            AlignProcrustes
│   │   ├── align_cayley.py                CayleyOrthogonal (SO(d) rotation)
│   │   ├── fuse.py                        FuseGeometric
│   │   ├── context_film.py                FiLMLayer (feature-wise linear modulation)
│   │   ├── context_position_geometric.py  PositionGeometricContext (5-stream fusion)
│   │   ├── compose_quaternion.py          QuaternionCompose (Hamilton product)
│   │   ├── project_manifold.py            ManifoldProjection (h -> S^(d-1))
│   │   ├── attend_geometric.py            GeometricAttention (FiLM on Q,K)
│   │   ├── attend_content.py              ContentAttention (standard MHA)
│   │   ├── distinguish_nce_bank.py        GeoResidualBank (CLIP-style contrastive)
│   │   ├── layer_geometric.py             GeometricTransformerLayer (one layer)
│   │   ├── transformer_geometric.py       GeometricTransformer + factory functions
│   │   └── geometric_transformer.py       Backward-compatible re-export shim
│   ├── observer.py                    Stage interfaces (Input, Association, ...)
│   ├── arbitrary_feature.py           GeometricPipeline (token -> geo feature)
│   ├── esm2_geometric.py             ESM-2 protein geometric pipeline
│   ├── layer.py                       ConstellationLayer (one depth)
│   └── backbone.py                    GeometricBackbone (multi-depth)
│
├── training/                      Training orchestration
│   └── curriculum.py                  CurriculumTrainer (episodic crystallization)
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
│   ├── constellation_encoder.py       Reference: MagnitudeFlow + ConstellationObserver
│   ├── conv_encoder.py                Conv2d -> (emb, magnitude)
│   ├── conv_svd_encoder.py            Conv2d + SVD features
│   ├── conv_scatter_encoder.py        Conv2d + Scattering transform
│   ├── spectral_encoder.py            Spectral features
│   └── transformer_svd_encoder.py     Transformer + SVD
│
├── analysis/                      Diagnostic tools
│   └── geometric.py                   cv_metric, knn_accuracy, analyze_svd_model
│
├── utils/                         Engineering infrastructure
│   ├── kernel.py                      Triton SVD N=2,3, gram_eigh, Procrustes math
│   ├── cuda/                          CuPy/NVRTC eigendecomposition kernels
│   ├── triton/                        Generated Triton kernels (experimental)
│   └── memory.py                      EmbeddingBuffer
│
└── SYSTEM.md                      Comprehensive system reference
```

## Quick Start

### Geometric Transformer

The main working model. CM-validated dual-stream with constellation routing and optional FlowEnsemble.

```python
from geolip_core.pipeline.components import (
    GeometricTransformer, geo_transformer_small)

# Factory: d=256, 8 heads, 4 layers, 16 anchors
model = geo_transformer_small('my_model', n_layers=4)
model.network_to(device='cuda', strict=False)

# Forward
x = torch.randn(B, L, 256, device='cuda')
out = model(x)                              # (B, L, 256)
out, geo_states = model(x, return_geo_state=True)  # + per-layer geometric state

# Training with observer loss
model.invalidate_caches()
model.precompute_cm_gates()     # cache CM det + Cayley solve (every 10-20 batches)
output = model.forward_paired(x1, x2)
loss, ld = model.compute_loss(output, targets, head=classifier_head)
loss.backward()

# Geometric regularization (call every step)
geo_losses = model.geometric_losses()
total = task_loss + geo_losses['geo_total']
```

Available factories: `geo_transformer_esm2` (d=1280, ESM-2 scale), `geo_transformer_small` (d=256, prototyping), `geo_transformer_vision` (d=384, vision patches).

### Composable Pipeline (geofractal router)

```python
from geolip_core.pipeline.arbitrary_feature import GeometricPipeline
from geolip_core.pipeline.components import CurateCMGate

# Build pipeline: (B, 5, 512) -> (B, feature_dim)
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

`geolip_core.linalg` is a drop-in replacement for `torch.linalg`. Our implementations override where we have something better. Everything else transparently proxies to PyTorch.

```python
import geolip_core.linalg as LA

# Our implementations (auto-dispatch to best available)
vals, vecs = LA.eigh(A)           # FL pipeline n<=12, cuSOLVER n>12
U, S, Vh = LA.svd(A)              # Triton n=2,3 -> FL n<=12 -> cuSOLVER

# These pass through to torch.linalg (zero overhead)
x = LA.solve(A, b)
L = LA.cholesky(A)
n = LA.norm(x)

# Configuration
LA.backend.status()               # print available features
LA.backend.use_fl_eigh = False    # disable FL, use cuSOLVER everywhere
```

### Core Modules Directly

```python
from geolip_core.core.associate.constellation import ConstellationObserver
from geolip_core.core.curate.patchwork import Patchwork, AnchorPush

# No router, no cache -- just PyTorch modules
obs = ConstellationObserver(dim=256, n_anchors=32, n_comp=8, d_comp=32)
result = obs.observe(embedding)   # dict: embedding, triangulation, assignment, patchwork, ...

# Non-gradient anchor repositioning
push = AnchorPush('momentum', n_anchors=32, dim=256, alpha=0.05, beta=0.02)
push.push(obs, embedding_buffer, label_buffer)
```

### Curriculum Training (Episodic Crystallization)

```python
from geolip_core.training import CurriculumTrainer

trainer = CurriculumTrainer(model, config={
    'manifold_dim': 256, 'n_anchors': 128, 'n_classes': 100,
    'project_fn': my_project_fn,          # model, inputs -> (B, 256)
    'get_constellation_fn': get_const,     # model -> Constellation
    'set_constellation_fn': set_const,     # model, Constellation -> None
    'train_fn': my_train_fn,               # model, loader, epochs, loss_config -> metrics
})

# Stage 0: CE only -> crystallize constellation from stable embeddings
# Stage 1: Full observer losses with crystallized constellation
history = trainer.train_curriculum(loader, stages=[
    {'epochs': 50, 'losses': ['ce']},
    {'epochs': 50, 'losses': 'all'},
])
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

### Geometric Transformer Components

Extracted from the CM-validated dual-stream geometric transformer. Each is a standalone file.

| Component | Parent | Purpose |
|---|---|---|
| `CMValidatedGate` | `nn.Module` | Anchor gating via Cayley-Menger validity (cached precompute) |
| `GeoResidualBank` | `nn.Module` | Cross-stream contrastive memory bank (CLIP-style) |
| `FiLMLayer` | `TorchComponent` | Feature-wise Linear Modulation (near-identity init) |
| `CayleyOrthogonal` | `TorchComponent` | SO(d) rotation via Cayley map (det=1 always) |
| `QuaternionCompose` | `TorchComponent` | Four-arm Hamilton product composition |
| `ManifoldProjection` | `TorchComponent` | Project hidden states to S^(d-1) |
| `PositionGeometricContext` | `TorchComponent` | 5-stream fusion -> FiLM context |
| `GeometricAttention` | `TorchComponent` | FiLM on Q,K (geometry routes attention, V stays pure) |
| `ContentAttention` | `TorchComponent` | Standard self-attention (Stream A, no geometry) |
| `GeometricTransformerLayer` | `BaseTower` | One layer: project -> associate -> gate -> curate -> dual-stream -> compose |
| `GeometricTransformer` | `BaseTower` | Full model: layer stack + cross-layer rotation + NCE bank |

### Geometric Linear Algebra (`linalg/`)

Drop-in `torch.linalg` replacement with geometric optimizations. Anything not overridden proxies transparently to PyTorch.

| Function | Implementation | Performance |
|---|---|---|
| `eigh(A)` | FL Hybrid (n<=12), cuSOLVER fallback | 84/84 purity, 40x less memory, zero graph breaks |
| `svd(A)` | Triton N=2,3 -> FL eigh N<=12 -> cuSOLVER | 3,850x at N=2, compilable |
| `newton_schulz_invsqrt(G)` | Pure bmm iteration | Zero eigensolvers, quadratic convergence |
| `procrustes(src, tgt)` | Subspace-preserving rotation | 1.000 NN agreement at N=32-128 |
| Everything else | `torch.linalg` passthrough | Zero overhead |

Backend auto-detection with single warning on first fallback:

```python
from geolip_core.linalg import backend
backend.status()
# geolip_core.linalg backend:
#   CUDA:       yes
#   Triton:     3.6.0
#   FL eigh:    enabled
#   Triton SVD: enabled
#   GPU:        NVIDIA RTX PRO 6000 Blackwell Server Edition
```

**FL Hybrid Eigendecomposition** -- Faddeev-LeVerrier characteristic polynomial -> Laguerre root-finding -> Newton-Schulz orthogonalization -> Rayleigh quotient refinement. Wins 84/84 mathematical purity metrics against cuSOLVER across n=3-12. Zero graph breaks under `torch.compile(fullgraph=True)`. 40x less memory.

**CUDA Eigendecomposition Kernel** -- Per-matrix CUDA kernel via CuPy/NVRTC. One thread per matrix, entire pipeline in registers. Flat batch scaling -- 1.73x cuSOLVER at B=16,384, runs where cuSOLVER OOMs at B=32,768.

Benchmarked on NVIDIA RTX PRO 6000 Blackwell (B=4,096, n=6):

| Method | Time | vs cuSOLVER | Memory |
|---|---|---|---|
| cuSOLVER | 241 us | 1.00x | 1,099 MB |
| FL Precise compiled | 350 us | 0.69x | 32 MB |
| FL Precise + CUDA Graph | 287 us | 0.84x | 32 MB |
| CUDA kernel (n=6, B=16K) | 429 us | 1.73x | 0.7 MB |
| CUDA kernel (n=3, B=4K) | 45 us | 3.06x | 0.7 MB |

## Empirical Results

| System | Metric | Value |
|---|---|---|
| **GeoTransformer Redux v8** | CIFAR-100 accuracy | 61%, 6.8M params, 4 layers |
| | CV_anc | 0.212 (in pentachoron band) |
| | Active anchors | 126/128 |
| | Speed | 19s/epoch on RTX PRO 6000 Blackwell |
| **Ryan Spearman** | ProteinGym benchmark (84 unseen assays) | rho=0.309 (Procrustes matched) |
| | Wins vs GeoQuat | 76/84 (90%) |
| | Beta-lactamase | rho=0.550 approaching SOTA from zero training data |
| **FL Eigh** | Mathematical purity | 84/84 |
| | vs cuSOLVER speed | 1.73x faster (CUDA kernel) |
| | vs cuSOLVER memory | 40x less |
| **Procrustes survey** | Models analyzed | 17 |
| | QK eigenvalue lock | 0.500 universal |
| **GEOLIP-Bertenstein** | Retrieval | Perfect on 40K+ pairs, 1 epoch, 1 layer |

## Empirical Constants

| Constant | Value | Observed across |
|---|---|---|
| CV pentachoron band | 0.20-0.23 | 17+ architectures, all modalities |
| Binding/separation boundary | 0.29154 / 0.70846 | MinimalShunts, CLIP, T5, alpha convergence |
| Effective geometric dimension | 16 (S^15) | Validated in patchwork and anchor experiments |
| Irreducible CV minimum | 0.125 | Theoretical lower bound on sphere |
| Cross-modal QK eigenvalue lock | 0.500 | Universal across 17+ models |

## Requirements

```
torch >= 2.0
geofractal @ git+https://github.com/AbstractEyes/geofractal.git
```

Optional:
- `triton >= 2.1` -- fused SVD kernels for N=2,3
- `cupy-cuda12x` -- CUDA eigendecomposition kernels via NVRTC
- `kymatio` -- scattering transforms

## Ecosystem

- [geofractal](https://github.com/AbstractEyes/geofractal) -- Router/tower/component composition framework
- [wide-compiler](https://github.com/AbstractEyes/wide-compiler) -- N-model batched fusion via grouped operations
- [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) -- Full GeoLIP package
- [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) -- Multi-expert geometric fusion
- [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) -- Cross-model alignment study
- [ryan-spearman](https://huggingface.co/AbstractPhil/ryan-spearman-prepared-features) -- Variant effect prediction heads + features
- [FL Eigh Article](https://huggingface.co/blog/AbstractPhil/linalg-eigh-rehaul-ft1) -- Compilable eigendecomposition beating cuSOLVER
- [SVD Kernel Article](https://huggingface.co/blog/AbstractPhil/svd-triton-kernel-optimization) -- Engineering specification
- [SVD Experiment Journey](https://huggingface.co/blog/AbstractPhil/svd-experiment-journey) -- Development map

See `geolip_core/SYSTEM.md` for the comprehensive system reference.

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil). Apache 2.0.*
