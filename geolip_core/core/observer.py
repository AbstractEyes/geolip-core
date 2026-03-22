"""
GeoLIP Observer Framework — the six-stage observation paradigm.

Input       → raw signal → embedding on S^(d-1)
Mutation    → transform position on the manifold
Association → measure relationships to reference frame
Curation    → select what matters from associations
Distinction → task-specific output from curated features
Loss        → standalone recipes (in losses.py, not here)

GeoLIP is the composition loop. It chains stages, runs them,
and presents the result. The stages are independently swappable.

The canonical pipeline:
  Input → Association → Mutation → Curation → Association → Curation → Distinction

But GeoLIP doesn't enforce order. You register stages, you define the
pipeline in your forward method. GeoLIP holds them and exposes them.

Usage:
    from core.observer import Input, Mutation, Association, Curation, Distinction, GeoLIP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# STAGE INTERFACES
# ══════════════════════════════════════════════════════════════════

class Input(nn.Module):
    """Raw signal → embedding on S^(d-1).

    Consumes whatever the raw input is (pixels, tokens, scattering
    coefficients, frequency bands) and produces an L2-normalized
    embedding. The output lives on the manifold. Everything downstream
    assumes this.

    Subclass must implement:
        encode(x) → (B, dim) unnormalized features
        dim: output embedding dimension
    """

    @property
    def dim(self):
        """Output embedding dimension."""
        raise NotImplementedError

    def encode(self, x):
        """Raw input → unnormalized features. Subclass implements this.

        Returns:
            (B, dim) features — NOT yet L2-normalized
        """
        raise NotImplementedError

    def forward(self, x):
        """Raw input → L2-normalized embedding on S^(d-1).

        Also returns raw magnitude before normalization, since
        MagnitudeFlow and other transforms may need it.

        Returns:
            emb: (B, dim) on S^(d-1)
            magnitude: (B, 1) pre-normalization norm
        """
        feat = self.encode(x)
        magnitude = feat.norm(dim=-1, keepdim=True)
        emb = F.normalize(feat, dim=-1)
        return emb, magnitude


class Mutation(nn.Module):
    """Transform position on the manifold. S^(d-1) → S^(d-1).

    Receives an embedding that's already on the sphere and moves it.
    Flow, relay, projection, correction — anything that changes WHERE
    the embedding sits without changing what it represents.

    Subclass must implement:
        mutate(emb, **context) → (B, dim) on S^(d-1)
    """

    def mutate(self, emb, **context):
        """Move embedding on the manifold.

        Args:
            emb: (B, D) on S^(d-1)
            **context: additional info (triangulation, magnitude, etc.)

        Returns:
            (B, D) on S^(d-1)
        """
        raise NotImplementedError

    def forward(self, emb, **context):
        return self.mutate(emb, **context)


class Association(nn.Module):
    """Measure relationships to a reference frame.

    Takes an embedding on S^(d-1) and produces a distance profile,
    soft assignment, or any relational measurement against a structure
    (anchors, prototypes, centroids, frequency bands, etc.).

    This is the act of observing — converting position into relationship.

    Subclass must implement:
        associate(emb, **context) → dict with at minimum 'distances' key
        frame_dim: dimension of the association output (number of reference points)
    """

    @property
    def frame_dim(self):
        """Number of reference points in the frame."""
        raise NotImplementedError

    def associate(self, emb, **context):
        """Measure relationships between embedding and reference frame.

        Args:
            emb: (B, D) on S^(d-1)

        Returns:
            dict with association outputs. Must contain 'distances'.
        """
        raise NotImplementedError

    def forward(self, emb, **context):
        return self.associate(emb, **context)


class Curation(nn.Module):
    """Select what matters from associations.

    Takes the raw association output (distances, assignments) and
    produces an interpreted feature vector. Patchwork is one curation.
    Gram matrices, SVD signatures, channel selection — all curations.

    Subclass must implement:
        curate(association_output, **context) → (B, feature_dim) features
        feature_dim: output dimension
    """

    @property
    def feature_dim(self):
        """Output feature dimension."""
        raise NotImplementedError

    def curate(self, association_output, **context):
        """Interpret association output into features.

        Args:
            association_output: dict from Association.associate()

        Returns:
            (B, feature_dim) curated features
        """
        raise NotImplementedError

    def forward(self, association_output, **context):
        return self.curate(association_output, **context)


class Distinction(nn.Module):
    """Task-specific output from curated features.

    Classification head, generation head, retrieval projection —
    whatever the task needs. This is the only stage that knows
    about the downstream task.

    Subclass must implement:
        distinguish(features, **context) → task output
    """

    def distinguish(self, features, **context):
        """Produce task-specific output.

        Args:
            features: (B, feature_dim) from curation

        Returns:
            task output (logits, embeddings, etc.)
        """
        raise NotImplementedError

    def forward(self, features, **context):
        return self.distinguish(features, **context)


# ══════════════════════════════════════════════════════════════════
# GEOLIP — the composition loop
# ══════════════════════════════════════════════════════════════════

class GeoLIP(nn.Module):
    """The observation loop. Binds stages, holds them, presents them.

    GeoLIP doesn't enforce pipeline order. It's a container that holds
    named stages of any type and makes them accessible. Your forward
    method defines the actual pipeline.

    For simple cases, use register methods to add stages and access
    them by name. For complex multi-tier pipelines, subclass GeoLIP
    and define your own forward.

    GeoLIP can also be ignored entirely. Use stages directly.

    Args:
        dim: embedding dimension on S^(dim-1)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Stage containers — ModuleDict for proper parameter registration
        self.inputs = nn.ModuleDict()
        self.mutations = nn.ModuleDict()
        self.associations = nn.ModuleDict()
        self.curations = nn.ModuleDict()
        self.distinctions = nn.ModuleDict()

    def add_input(self, name, stage):
        """Register an Input stage."""
        self.inputs[name] = stage
        return self

    def add_mutation(self, name, stage):
        """Register a Mutation stage."""
        self.mutations[name] = stage
        return self

    def add_association(self, name, stage):
        """Register an Association stage."""
        self.associations[name] = stage
        return self

    def add_curation(self, name, stage):
        """Register a Curation stage."""
        self.curations[name] = stage
        return self

    def add_distinction(self, name, stage):
        """Register a Distinction stage."""
        self.distinctions[name] = stage
        return self

    @property
    def feature_dim(self):
        """Sum of all curation feature dimensions."""
        return sum(c.feature_dim for c in self.curations.values())

    def get_stage(self, category, name):
        """Retrieve a stage by category and name.

        Args:
            category: 'input', 'mutation', 'association', 'curation', 'distinction'
            name: stage name

        Returns:
            the stage module
        """
        containers = {
            'input': self.inputs,
            'mutation': self.mutations,
            'association': self.associations,
            'curation': self.curations,
            'distinction': self.distinctions,
        }
        return containers[category][name]