# geolip-core

Geometric observer framework for deep learning. Built on [geofractal](https://github.com/AbstractEyes/geofractal)'s router infrastructure.

Part of the [GeoLIP](https://github.com/AbstractEyes/glip-autoencoder) ecosystem.

## Install

```bash
pip install "git+https://github.com/AbstractEyes/geolip-core.git"
```

## The Six-Stage Paradigm

Every GeoLIP pipeline decomposes into six stages:

| Stage | What it does | Interface |
|---|---|---|
| **Input** | Raw signal → embedding on S^(d-1) | `encode(x) → features` |
| **Mutation** | Transform position on the manifold | `mutate(emb) → emb` |
| **Association** | Measure relationships to reference frame | `associate(emb) → dict` |
| **Curation** | Select what matters from associations | `curate(assoc) → features` |
| **Distinction** | Task-specific output | `distinguish(features) → output` |
| **Loss** | Standalone recipes | functions in `losses.py` |

The canonical pipeline is recursive:

```
Input → Association → Mutation → Curation → Association → Curation → Distinction
```

Each stage is a `TorchComponent` with identity (name, uuid), lifecycle hooks, and device affinity — inherited from geofractal's component system. Stages work standalone as regular `nn.Module`s or composed into a `GeoLIP` tower.

`GeoLIP` is a `BaseTower`. Stages are attached as named components via `attach()` and accessed through `self['name']`. Forward defines the observation pipeline. The paradigm guides structure — it doesn't force ceremony.

## Constellation Implementations

The constellation is one concrete set of stage implementations:

| Class | Stage | Description |
|---|---|---|
| `ConvEncoder` | Input | 8-layer conv → S^(d-1) |
| `ConstellationRelay` | Mutation | Per-token triangulation + gated residual, O(S) |
| `FlowAttention` | Mutation | ODE flow in tangent space (historical) |
| `MagnitudeFlow` | Mutation | Relay-stack per-compartment magnitude prediction |
| `ConstellationAssociation` | Association | Triangulate against learned anchors |
| `Patchwork` | Curation | Round-robin compartmentalized interpretation |
| `ConstellationCuration` | Curation | Patchwork + bridge over association output |
| `ClassificationHead` | Distinction | MLP classification head |
| `observer_loss` | Loss | Standalone geometric + internal self-organization recipe |

`ConstellationObserver` composes `ConstellationAssociation` + `ConstellationCuration` with `observe()` and `observe_paired()`.

`ConstellationEncoder` is a `GeoLIP` tower with constellation stages pre-attached — the full-stack geometric pipeline, modality-agnostic.

## Quick Start

### Individual stages

```python
from geolip_core.core import ConstellationAssociation, ConstellationCuration
import torch, torch.nn.functional as F

assoc = ConstellationAssociation(dim=128, n_anchors=64)
curate = ConstellationCuration(n_anchors=64, dim=128)

emb = F.normalize(torch.randn(32, 128), dim=-1)
a_out = assoc(emb)                         # Association → distances, assignment, etc.
features = curate(a_out, emb=emb)          # Curation → (32, feature_dim)
```

### Composed observer

```python
from geolip_core.core import ConstellationObserver, observer_loss

obs = ConstellationObserver(dim=128, n_anchors=64)
out = obs.observe_paired(emb1, emb2)

# Standalone loss recipe (no CE, pure geometric self-organization)
loss, ld = observer_loss(out, obs.constellation.anchors, targets=labels)
```

### Full classification pipeline (new paradigm)

```python
from geolip_core.encoder import ConstellationEncoder, GeoLIPEncoder, ConvEncoder

# ConstellationEncoder — modality-agnostic, accepts any embedding on S^(d-1)
enc = ConstellationEncoder(dim=384, n_anchors=512, num_classes=100)
out = enc.forward_paired(emb1, emb2, raw_mag1, raw_mag2)
loss, ld = enc.compute_loss(out, targets)

# GeoLIPEncoder — any Input stage + ConstellationEncoder composed
model = GeoLIPEncoder(ConvEncoder(384), num_classes=100, n_anchors=512)
out = model.forward_paired(view1, view2)
loss, ld = model.compute_loss(out, targets)

# Optimizer with proper anchor weight-decay exclusion
optimizer = model.make_optimizer(lr=3e-4, weight_decay=0.05)
```

### Legacy pipeline (pre-paradigm, still works)

```python
from geolip_core.encoder import GeoLIPConvEncoder

model = GeoLIPConvEncoder(num_classes=100, output_dim=384, n_anchors=512)
out = model.forward_paired(view1, view2)
loss, ld = model.compute_loss(out, targets)
```

## Building New Stages

Implement the stage interfaces. Each stage is a `TorchComponent` — it has identity, lifecycle hooks, and works as a standalone `nn.Module`. Existing loss functions and downstream heads work with any implementation that follows the interface contracts.

```python
from geolip_core.core import Input, Association, Curation

class MyEncoder(Input):
    """Custom Input stage. encode() returns unnormalized features.
    forward() (inherited) handles L2 normalization + magnitude extraction."""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
        self.net = ...  # your backbone

    @property
    def dim(self): return self._dim

    def encode(self, x): return self.net(x)


class MyAssociation(Association):
    """Custom Association. Must return dict with 'distances' key."""
    def __init__(self, frame_size, **kwargs):
        super().__init__(**kwargs)
        self._frame_dim = frame_size

    @property
    def frame_dim(self): return self._frame_dim

    def associate(self, emb, **ctx):
        return {'distances': my_distance_fn(emb), 'cos_to_anchors': ...}


class MyCuration(Curation):
    """Custom Curation. Interprets association output into features."""
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self._feature_dim = out_dim
        self.mlp = ...

    @property
    def feature_dim(self): return self._feature_dim

    def curate(self, assoc_out, **ctx):
        return self.mlp(assoc_out['distances'])
```

### Composing into a GeoLIP tower

`GeoLIP` is a `BaseTower` from geofractal. Attach stages, define forward:

```python
from geolip_core.core import GeoLIP

class MyObserver(GeoLIP):
    def __init__(self, dim):
        super().__init__('my_observer', dim, strict=False)
        self.attach('assoc', MyAssociation(frame_size=64))
        self.attach('curate', MyCuration(out_dim=128))

    def forward(self, emb):
        a_out = self['assoc'](emb)
        features = self['curate'](a_out, emb=emb)
        return features
```

Access patterns inherited from `BaseTower`:

```python
obs['assoc']           # Named component
obs.has('curate')      # Check existence
obs.cache_set(k, v)    # Ephemeral tensor storage
obs.cache_clear()      # Clear after forward
```

## Architecture

```
geolip_core/
├── core/
│   ├── observer.py              # Stage interfaces (TorchComponent) + GeoLIP (BaseTower)
│   ├── constellation.py         # Constellation, ConstellationAssociation, ConstellationCuration, ConstellationObserver
│   ├── patchwork.py             # Patchwork, MagnitudeFlow, AnchorPush
│   ├── constellation_relay.py   # RelayLayer, ConstellationRelay
│   ├── constellation_route.py   # FlowAttention (historical)
│   ├── losses.py                # All losses + metrics (batched CV, NCE, bridge, etc.)
│   ├── activation.py            # SquaredReLU, StarReLU, make_activation
│   ├── memory.py                # EmbeddingBuffer
│   └── core.py                  # GeometricAutograd, param_count, model_summary
├── encoder/
│   ├── encoder_constellation.py # ConstellationEncoder (GeoLIP tower), ClassificationHead, GeoLIPEncoder
│   ├── encoder_conv.py          # ConvEncoder (Input stage), GeoLIPConvEncoder (legacy)
│   ├── encoder_scatterpoint.py  # (future)
│   ├── encoder_wavelet.py       # (future)
│   ├── encoder_transformer.py   # (future)
│   └── encoder_sha.py           # (future)
└── distillation/                # (future: alignment, bank, expert)
```

## Key Empirical Constants

- **CV pentachoron band**: 0.20–0.23 in trained models (natural basin ~0.24 on pure noise at d=128)
- **Binding/separation boundary**: 0.29154 / 0.70846 radians
- **Effective geometric dimension**: ~16 (S^15)
- **Relay fidelity**: 99.4% cosine preservation through 16 stacked layers
- **CV floor**: ~0.11 (hard geometric limit, cannot be pushed lower by any loss weight)

## Requirements

```
torch >= 2.0
geofractal (git+https://github.com/AbstractEyes/geofractal.git)
```

## Ecosystem

### Infrastructure

| Repository | Description |
|---|---|
| [geofractal](https://github.com/AbstractEyes/geofractal) | Router infrastructure — BaseTower, WideRouter, TorchComponent. The foundation this repo builds on. |
| [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) | Full GeoLIP package (PyPI: `geolip`). The parent ecosystem repo. |

### Constellation Models & Experiments

| Repository | Description |
|---|---|
| [geolip-constellation-core](https://huggingface.co/AbstractPhil/geolip-constellation-core) | HuggingFace model hub — constellation architecture documentation and pretrained checkpoints |
| [geolip-captionbert-8192](https://huggingface.co/AbstractPhil/geolip-captionbert-8192) | CaptionBERT — geometric caption encoder with Procrustes consensus distillation, 8192 anchors. Feature extraction pipeline with pentachoron structure. |
| [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) | Multi-expert geometric fusion transformer — BERT-large hub with frozen expert encoders (DINOv2, Whisper, ESM-2, CodeBERT) |

### CV & Geometric Analysis

| Repository | Description |
|---|---|
| [geolip-cv-experiments](https://huggingface.co/AbstractPhil/geolip-cv-experiments) | CV loss sweep — batched 141x speedup validation, weight/target/dimension sweeps |
| [geolip-cv-noise-analysis](https://huggingface.co/AbstractPhil/geolip-cv-noise-analysis) | Constellation relay preservation analysis — established attention weakness (cycle 1 to cycle N), validated relay geometric fidelity |
| [geolip-hypersphere-experiments](https://huggingface.co/AbstractPhil/geolip-hypersphere-experiments) | Spectral encoder test manifest — wavelet scattering (Mallat), Gabor filter banks, Radon/curvelet transforms as inputs to the constellation pipeline. Origin of the avg-pool spatial collapse finding. |
| [geolip-constellation-activations](https://huggingface.co/AbstractPhil/geolip-constellation-activations) | Cross-token relay sequence prototyping and routing optimization — benchmarking constellation relay across token interactions |

### Vision Architectures

| Repository | Description |
|---|---|
| [geolip-vit-dual-stream](https://huggingface.co/AbstractPhil/geolip-vit-dual-stream) | Dual-stream ViT — 11+ run research log. Dual InfoNCE, shared attention, mastery queue, BCE stream. Established that with cross-attention enabled, the geometric structure alone classifies the entire thing. |
| [geolip-vit-tri-stream](https://huggingface.co/AbstractPhil/geolip-vit-tri-stream) | Tri-stream ViT — evolution of dual-stream. Three processing paths: Stream A (CE), Stream B (BCE+NCE), GAL (geometric arbitration layer with KSimplex features + Procrustes anchor rotation). |
| [geolip-vit-zana](https://huggingface.co/AbstractPhil/geolip-vit-zana) | Zana ViT — PentaViT refactored into the GeoLIP paradigm. Custom automodel functional. Geometric constellation structure intact as standalone prototype, evolved from penta-vit-experiments. |
| [geolip-vit-x34](https://huggingface.co/AbstractPhil/geolip-vit-x34) | x34 ViT — 34-model soup with Procrustes alignment to constellation patchwork. Early attempt at multi-model geometric alignment, identified that Procrustes data was too premature for direct ViT training. |

### CLIP Geometric Distillation

| Repository | Description |
|---|---|
| [geolip-clip-vit-bigG-patch14-ctx576-seq77](https://huggingface.co/AbstractPhil/geolip-clip-vit-bigG-patch14-ctx576-seq77) | Memory-augmented CLIP ViT-bigG distillation — ModernBERT + Procrustes + pentachoron structure, 576 context / 77 sequence, for long-context text encoding targeting SDXL |
| [geolip-clip-vit-large-patch14-ctx576-seq77](https://huggingface.co/AbstractPhil/geolip-clip-vit-large-patch14-ctx576-seq77) | Memory-augmented CLIP ViT-L/14 distillation — 576 context / 77 sequence variant |
| [geolip-clip-vit-large-patch14-ctx576](https://huggingface.co/AbstractPhil/geolip-clip-vit-large-patch14-ctx576) | Memory-augmented CLIP ViT-L/14 distillation — 576 context variant |

### Diffusion & Generation

| Repository | Description |
|---|---|
| [geolip-spherical-diffusion-proto](https://huggingface.co/AbstractPhil/geolip-spherical-diffusion-proto) | Spherical flow-matching diffusion — geometric loss on S^(d-1), constellation-anchored generation on CIFAR-10 |
| [geolip-diffusion-proto](https://huggingface.co/AbstractPhil/geolip-diffusion-proto) | Flow-match relay diffusion — constellation relay integrated into diffusion pipeline. Geometric structure contributes ~6-7% to output. Automodel available. |

### Cross-Model Analysis

| Repository | Description |
|---|---|
| [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) | Procrustes alignment study — 17 models profiled (T5 family, BERT, CLIP, DINOv2, UNets, VAEs). Cross-modal QK eigenvalue lock at 0.500, VAE weights 70-76% alignable. |
| [geolip-procrustes](https://huggingface.co/AbstractPhil/geolip-procrustes) | GeoLIP-specific Procrustes data and alignment experiments (part of GEOLIP Research Concepts collection) |

### Evolutionary & Experimental

| Repository | Description |
|---|---|
| [geolip-genetic-inheritance](https://huggingface.co/AbstractPhil/geolip-genetic-inheritance) | Genetic inheritance experiment — anchor vectors heritable across generations, CV converges from ~1.7 (Gen 0) to ~0.33 (Gen 4), genetic diversity beats pure fitness selection |
| [geometric-experiment-history](https://huggingface.co/AbstractPhil/geometric-experiment-history) | **The complete project catalog** — 33 projects across 9 research areas: pentachoron mathematics, geometric vocabulary, classification architectures, language models, diffusion, feature extraction, consciousness research, scaling architecture, infrastructure |

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil). Complete experiment history and project catalog at [geometric-experiment-history](https://huggingface.co/AbstractPhil/geometric-experiment-history).*