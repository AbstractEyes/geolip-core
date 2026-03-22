# geolip-core

Geometric observer framework for deep learning.

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

The canonical pipeline:

```
Input → Association → Mutation → Curation → Association → Curation → Distinction
```

Each stage is a swappable `nn.Module` with a minimal interface. `GeoLIP` is the composition container — it holds named stages and makes them accessible. It doesn't enforce order.

## Constellation Implementations

The constellation is one concrete set of stage implementations:

| Class | Stage | Description |
|---|---|---|
| `ConvEncoder` | Input | 8-layer conv → S^(d-1) |
| `ConstellationRelay` | Mutation | Per-token triangulation + gated residual, O(S) |
| `FlowAttention` | Mutation | ODE flow in tangent space |
| `MagnitudeFlow` | Mutation | Relay-stack magnitude prediction |
| `ConstellationAssociation` | Association | Triangulate against learned anchors |
| `Patchwork` | Curation | Round-robin compartmentalized interpretation |
| `ConstellationCuration` | Curation | Patchwork + bridge over association output |
| `observer_loss` | Loss | Standalone geometric + internal self-organization recipe |

`ConstellationObserver` is a convenience composition of `ConstellationAssociation` + `ConstellationCuration`.

## Quick Start

```python
from geolip_core.core import ConstellationAssociation, ConstellationCuration
import torch, torch.nn.functional as F

# Individual stages
assoc = ConstellationAssociation(dim=128, n_anchors=64)
curate = ConstellationCuration(n_anchors=64, dim=128)

emb = F.normalize(torch.randn(32, 128), dim=-1)
a_out = assoc(emb)                         # Association → distances, assignment, etc.
features = curate(a_out, emb=emb)          # Curation → (32, feature_dim)
```

```python
from geolip_core.core import ConstellationObserver, observer_loss

# Composed observer (Association + Curation)
obs = ConstellationObserver(dim=128, n_anchors=64)
out = obs.observe_paired(emb1, emb2)

# Standalone loss recipe
loss, ld = observer_loss(out, obs.constellation.anchors, targets=labels)
```

```python
from geolip_core.encoder import GeoLIPConvEncoder

# Full classification pipeline
model = GeoLIPConvEncoder(num_classes=100, output_dim=384, n_anchors=512)
out = model.forward_paired(view1, view2)
loss, ld = model.compute_loss(out, targets)
```

## Building New Prototypes

Implement the stage interfaces. Plug them in. The existing loss functions and downstream heads work with any implementation:

```python
from geolip_core.core import Input, Association, Curation
import torch.nn.functional as F

class MyEncoder(Input):
    @property
    def dim(self): return 256
    def encode(self, x): return self.net(x)  # your backbone

class MyAssociation(Association):
    @property
    def frame_dim(self): return 32
    def associate(self, emb, **ctx): return {'distances': my_distances(emb)}

class MyCuration(Curation):
    @property
    def feature_dim(self): return 128
    def curate(self, assoc_out, **ctx): return self.mlp(assoc_out['distances'])
```

## Key Empirical Constants

- **CV pentachoron band**: 0.20–0.23 (universal across 17+ architectures)
- **Binding/separation boundary**: 0.29154 / 0.70846
- **Effective geometric dimension**: ~16 (S^15)
- **Relay fidelity**: 99.4% cosine preservation through 16 stacked layers

## Requirements

```
torch >= 2.0
```

## Ecosystem

- [glip-autoencoder](https://github.com/AbstractEyes/glip-autoencoder) — Full GeoLIP package (PyPI: `geolip`)
- [geolip-constellation-core](https://huggingface.co/AbstractPhil/geolip-constellation-core) — HuggingFace model
- [geolip-bertenstein](https://huggingface.co/AbstractPhil/geolip-bertenstein) — Multi-expert geometric fusion
- [procrustes-analysis](https://huggingface.co/AbstractPhil/procrustes-analysis) — Cross-model alignment study

---

*Research by [AbstractPhil](https://huggingface.co/AbstractPhil).*