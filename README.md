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

- [geofractal](https://github.com/AbstractEyes/geofractal) — Router infrastructure (BaseTower, WideRouter, TorchComponent)
- [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) — Full GeoLIP package (PyPI: `geolip`)
- [geolip-constellation-core](https://huggingface.co/AbstractPhil/geolip-constellation-core) — HuggingFace model + documentation
- [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) — Multi-expert geometric fusion
- [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) — Cross-model alignment study

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil).*