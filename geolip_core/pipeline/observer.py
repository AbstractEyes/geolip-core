"""
GeoLIP Observer — Six-Stage Geometric Observation Paradigm
=============================================================
Built on geofractal's router infrastructure.

Stages:
    Input       → raw signal → embedding on S^(d-1)
    Mutation    → transform position on the manifold
    Association → measure relationships to reference frame
    Curation    → select what matters from associations
    Distinction → task-specific output from curated features
    Loss        → standalone recipes (in losses.py, not here)

The canonical pipeline:
    Input → Association → Mutation → Curation → Association → Curation → Distinction

GeoLIP is the composition — a BaseTower that holds stages as named
components and defines the observation flow in forward(). The paradigm
is recursive: Association → Curation can repeat at multiple tiers.

Each stage is a TorchComponent — it has identity, lifecycle hooks,
device affinity, and works as a standalone nn.Module. GeoLIP
composition is optional. Use stages directly if you prefer.

Usage:
    from geolip_core.core.observer import Input, Association, Curation, GeoLIP

    # Stages work standalone
    assoc = MyAssociation('assoc', dim=384)
    out = assoc(emb)

    # Or compose into a GeoLIP tower
    lip = GeoLIP('observer', dim=384)
    lip.attach('assoc', assoc)
    lip.attach('curate', MyCuration('curate', dim=384))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# geofractal is optional — fall back to nn.Module if not installed
try:
    from geofractal.router.base_tower import BaseTower as _BaseTower
    from geofractal.router.components.torch_component import TorchComponent as _TorchComponent
    _HAS_GEOFRACTAL = True
except ImportError:
    _HAS_GEOFRACTAL = False

    class _TorchComponent(nn.Module):
        """Fallback when geofractal is not installed."""
        def __init__(self, name=None, **kwargs):
            super().__init__()
            self._component_name = name or self.__class__.__name__

    class _BaseTower(nn.Module):
        """Fallback when geofractal is not installed."""
        def __init__(self, name=None, **kwargs):
            super().__init__()
            self._tower_name = name or self.__class__.__name__
            self._components = nn.ModuleDict()

        def attach(self, name, module):
            if isinstance(module, nn.Module):
                self._components[name] = module
            return self

        def has(self, name):
            return name in self._components

        def __getitem__(self, key):
            return self._components[key]

TorchComponent = _TorchComponent
BaseTower = _BaseTower


# ══════════════════════════════════════════════════════════════════
# STAGE INTERFACES
# ══════════════════════════════════════════════════════════════════

class Input(TorchComponent):
    """Raw signal → embedding on S^(d-1).

    Subclass must implement:
        encode(x) → (B, dim) unnormalized features
        dim (property) → int

    forward() auto-handles L2 normalization and magnitude extraction.
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name or self.__class__.__name__, **kwargs)

    @property
    def dim(self):
        raise NotImplementedError

    def encode(self, x):
        """Raw input → unnormalized features. Subclass implements this."""
        raise NotImplementedError

    def forward(self, x):
        """Raw input → (emb on S^(d-1), raw magnitude)."""
        feat = self.encode(x)
        magnitude = feat.norm(dim=-1, keepdim=True)
        emb = F.normalize(feat, dim=-1)
        return emb, magnitude


class Mutation(TorchComponent):
    """Transform position on the manifold. S^(d-1) → S^(d-1).

    Subclass must implement:
        mutate(emb, **context) → (B, D) on S^(d-1)
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name or self.__class__.__name__, **kwargs)

    def mutate(self, emb, **context):
        raise NotImplementedError

    def forward(self, emb, **context):
        return self.mutate(emb, **context)


class Association(TorchComponent):
    """Measure relationships to a reference frame.

    Subclass must implement:
        associate(emb, **context) → dict (must contain 'distances')
        frame_dim (property) → int
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name or self.__class__.__name__, **kwargs)

    @property
    def frame_dim(self):
        raise NotImplementedError

    def associate(self, emb, **context):
        raise NotImplementedError

    def forward(self, emb, **context):
        return self.associate(emb, **context)


class Curation(TorchComponent):
    """Interpret associations into features.

    Subclass must implement:
        curate(association_output, **context) → (B, feature_dim)
        feature_dim (property) → int
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name or self.__class__.__name__, **kwargs)

    @property
    def feature_dim(self):
        raise NotImplementedError

    def curate(self, association_output, **context):
        raise NotImplementedError

    def forward(self, association_output, **context):
        return self.curate(association_output, **context)


class Distinction(TorchComponent):
    """Task-specific output from curated features.

    Subclass must implement:
        distinguish(features, **context) → task output
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name or self.__class__.__name__, **kwargs)

    def distinguish(self, features, **context):
        raise NotImplementedError

    def forward(self, features, **context):
        return self.distinguish(features, **context)


# ══════════════════════════════════════════════════════════════════
# GEOLIP — observation tower
# ══════════════════════════════════════════════════════════════════

class GeoLIP(BaseTower):
    """Geometric observation tower.

    A BaseTower configured for the observation paradigm. Stages are
    attached as named components. Forward defines the pipeline.

    GeoLIP doesn't enforce a specific pipeline. Attach your stages,
    define forward(). The six-stage paradigm guides structure — it's
    the blueprint, not the constraint.

    Access patterns (inherited from BaseTower):
        lip['association']  → named component
        lip[0]              → stage by index (if using stages)
        lip.has('curation') → check existence
        for stage in lip:   → iterate stages

    Storage (inherited from BaseRouter):
        lip.attach('name', module)   → nn.Module component
        lip.attach('config', dict)   → non-module object
        lip.cache_set('key', tensor) → ephemeral tensor

    Args:
        name: tower name
        dim: embedding dimension on S^(dim-1)
    """

    def __init__(self, name, dim, **kwargs):
        super().__init__(name, **kwargs)
        self.dim = dim

    def forward(self, *args, **kwargs):
        """Subclass defines the observation pipeline.

        Typical pattern:
            emb → self['association'](emb)
                → self['mutation'](emb, ...)
                → self['curation'](assoc_out, ...)
                → self['distinction'](features)
        """
        raise NotImplementedError(
            "GeoLIP.forward() must be defined by the subclass or composition. "
            "Attach stages with self.attach() and define the pipeline in forward()."
        )