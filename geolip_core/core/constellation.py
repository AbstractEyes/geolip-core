"""
Constellation — anchor frame decomposed into stages.

Constellation:              The anchor primitive. Learned points on S^(d-1).
ConstellationAssociation:   Association stage. Triangulates against anchors.
ConstellationCuration:      Curation stage. Patchwork interpretation + bridge.
ConstellationObserver:      Convenience composition of association + curation.

Usage:
    # Use stages directly
    assoc = ConstellationAssociation(dim=384, n_anchors=64)
    curate = ConstellationCuration(n_anchors=64, dim=384)
    a_out = assoc(emb)
    features = curate(a_out)

    # Or use the composed observer
    obs = ConstellationObserver(dim=384, n_anchors=64)
    out = obs.observe(emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observer import Association, Curation


# ── ANCHOR INITIALIZATION ──

def init_anchors_xavier(n, d):
    w = torch.empty(n, d)
    nn.init.xavier_normal_(w)
    return F.normalize(w, dim=-1)


def init_anchors_orthogonal(n, d):
    if n <= d:
        Q, _ = torch.linalg.qr(torch.randn(d, n))
        return Q.T.contiguous()
    else:
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        return torch.cat([Q.T, F.normalize(torch.randn(n - d, d), dim=-1)], dim=0)


def init_anchors_repulsion(n, d, iters=200, lr=0.05):
    vecs = F.normalize(init_anchors_orthogonal(n, d), dim=-1)
    for _ in range(iters):
        sim = vecs @ vecs.T
        sim.fill_diagonal_(-2.0)
        vecs = F.normalize(vecs - lr * vecs[sim.argmax(dim=1)], dim=-1)
    return vecs


INIT_METHODS = {
    'xavier': init_anchors_xavier,
    'orthogonal': init_anchors_orthogonal,
    'repulsion': init_anchors_repulsion,
}


# ── CONSTELLATION — the anchor frame primitive ──

class Constellation(nn.Module):
    """Learned anchors on S^(d-1). Triangulates input embeddings.

    This is the primitive — the reference frame itself. Use it through
    ConstellationAssociation for the stage interface, or call
    triangulate() directly.

    Args:
        n_anchors: number of anchor points
        dim: embedding dimension
        anchor_drop: probability of dropping each anchor during training
        anchor_init: initialization strategy
    """

    def __init__(self, n_anchors, dim, anchor_drop=0.0, anchor_init='repulsion'):
        super().__init__()
        self.anchors = nn.Parameter(INIT_METHODS[anchor_init](n_anchors, dim))
        self.anchor_drop = anchor_drop
        self.n_anchors = n_anchors
        self.dim = dim

    def triangulate(self, emb, training=False):
        anchors = F.normalize(self.anchors, dim=-1)
        if training and self.anchor_drop > 0:
            mask = torch.rand(anchors.shape[0], device=anchors.device) > self.anchor_drop
            if mask.sum() < 2:
                mask[:2] = True
            anchors = anchors[mask]
            cos = emb @ anchors.T
            tri = 1.0 - cos
            _, nl = cos.max(dim=-1)
            return tri, mask.nonzero(as_tuple=True)[0][nl]
        cos = emb @ anchors.T
        tri = 1.0 - cos
        _, nearest = cos.max(dim=-1)
        return tri, nearest

    def forward(self, emb, training=False):
        return self.triangulate(emb, training)


# ── CONSTELLATION ASSOCIATION — the act of observing ──

class ConstellationAssociation(Association):
    """Association through constellation anchors.

    Triangulates embeddings against learned anchors on S^(d-1).
    Produces distances, cosine similarities, soft assignment, and
    nearest anchor index.

    Output dict:
        'distances':      (B, A) cosine distances (1 - cos)
        'cos_to_anchors': (B, A) raw cosine similarities
        'assignment':     (B, A) soft assignment via temperature-scaled softmax
        'nearest':        (B,) closest anchor index

    Args:
        dim: embedding dimension
        n_anchors: number of anchors
        anchor_drop: training anchor dropout
        anchor_init: initialization strategy
        assign_temp: softmax temperature for soft assignment
    """

    def __init__(self, dim=256, n_anchors=128, anchor_drop=0.15,
                 anchor_init='repulsion', assign_temp=0.1):
        super().__init__()
        self.assign_temp = assign_temp
        self.constellation = Constellation(n_anchors, dim, anchor_drop, anchor_init)

    @property
    def frame_dim(self):
        return self.constellation.n_anchors

    def associate(self, emb, **context):
        anchors_n = F.normalize(self.constellation.anchors, dim=-1)
        cos = emb @ anchors_n.T
        tri = 1.0 - cos
        _, nearest = cos.max(dim=-1)
        soft_assign = F.softmax(cos / self.assign_temp, dim=-1)

        # Apply magnitude weighting if provided
        mag = context.get('mag', None)
        distances_weighted = tri * mag if mag is not None else tri

        return {
            'distances': tri,
            'distances_weighted': distances_weighted,
            'cos_to_anchors': cos,
            'assignment': soft_assign,
            'nearest': nearest,
        }


# ── CONSTELLATION CURATION — interpretation of associations ──

class ConstellationCuration(Curation):
    """Curation through patchwork compartments + bridge.

    Reads the association output (distance profile) through round-robin
    patchwork compartments. Also produces a bridge prediction —
    patchwork's estimate of the constellation's soft assignment.

    The output features are: assignment + patchwork + embedding.
    This concatenation gives downstream heads access to all three
    levels of the observation.

    Output:
        (B, feature_dim) curated features

    Side output (accessible via curate_full):
        'patchwork': (B, pw_dim) compartment features
        'bridge': (B, A) bridge prediction
        'features': (B, feature_dim) full concatenated features

    Args:
        n_anchors: must match the association's frame_dim
        dim: embedding dimension (for feature concatenation)
        n_comp: patchwork compartments
        d_comp: hidden dim per compartment
        activation: activation function name
    """

    def __init__(self, n_anchors=128, dim=256, n_comp=8, d_comp=64,
                 activation='squared_relu'):
        super().__init__()
        self.dim = dim
        self.n_anchors = n_anchors

        from .patchwork import Patchwork
        self.patchwork = Patchwork(n_anchors, n_comp, d_comp, activation)
        pw_dim = self.patchwork.output_dim

        self.bridge = nn.Sequential(nn.Linear(pw_dim, n_anchors))

        # features = assignment + patchwork + embedding
        self._feature_dim = n_anchors + pw_dim + dim

    @property
    def feature_dim(self):
        return self._feature_dim

    def curate(self, association_output, emb=None, **context):
        """Interpret association into features.

        Args:
            association_output: dict from ConstellationAssociation.associate()
            emb: (B, D) the embedding — needed for feature concatenation

        Returns:
            (B, feature_dim) curated features
        """
        out = self.curate_full(association_output, emb=emb, **context)
        return out['features']

    def curate_full(self, association_output, emb=None, **context):
        """Full curation with all intermediate outputs.

        Returns:
            dict with 'patchwork', 'bridge', 'features'
        """
        distances = association_output['distances_weighted']
        assignment = association_output['assignment']

        pw = self.patchwork(distances)
        bridge = self.bridge(pw)

        parts = [assignment, pw]
        if emb is not None:
            parts.append(emb)
        features = torch.cat(parts, dim=-1)

        return {
            'patchwork': pw,
            'bridge': bridge,
            'features': features,
        }


# ── CONSTELLATION OBSERVER — composed association + curation ──

class ConstellationObserver(nn.Module):
    """Convenience composition of ConstellationAssociation + ConstellationCuration.

    For use with GeoLIP or standalone. Provides observe() and observe_paired()
    that run association → curation and return a merged output dict.

    This is NOT a base class or interface — it's a concrete composition
    of two stages that's useful enough to name.

    Args:
        dim: embedding dimension on S^(dim-1)
        n_anchors: anchor points
        n_comp: patchwork compartments
        d_comp: hidden dim per compartment
        anchor_drop: training anchor dropout
        anchor_init: anchor initialization strategy
        activation: activation function name
        assign_temp: assignment softmax temperature
    """

    def __init__(self, dim=256, n_anchors=128, n_comp=8, d_comp=64,
                 anchor_drop=0.15, anchor_init='repulsion',
                 activation='squared_relu', assign_temp=0.1):
        super().__init__()

        self.association = ConstellationAssociation(
            dim=dim, n_anchors=n_anchors, anchor_drop=anchor_drop,
            anchor_init=anchor_init, assign_temp=assign_temp,
        )
        self.curation = ConstellationCuration(
            n_anchors=n_anchors, dim=dim, n_comp=n_comp,
            d_comp=d_comp, activation=activation,
        )

        self.register_buffer('anchor_classes', torch.zeros(n_anchors, dtype=torch.long))

    @property
    def constellation(self):
        """Access the underlying Constellation primitive."""
        return self.association.constellation

    @property
    def patchwork(self):
        """Access the underlying Patchwork."""
        return self.curation.patchwork

    @property
    def feature_dim(self):
        return self.curation.feature_dim

    def observe(self, emb, **context):
        """Observe a single embedding.

        Returns dict with all association + curation outputs.
        """
        a_out = self.association(emb, **context)
        c_out = self.curation.curate_full(a_out, emb=emb, **context)

        return {
            'embedding': emb,
            'features': c_out['features'],
            'triangulation': a_out['distances'],
            'cos_to_anchors': a_out['cos_to_anchors'],
            'nearest': a_out['nearest'],
            'assignment': a_out['assignment'],
            'patchwork': c_out['patchwork'],
            'bridge': c_out['bridge'],
        }

    def observe_paired(self, emb1, emb2, mag1=None, mag2=None, **context):
        """Observe two views for contrastive training.

        Returns dict with outputs from both views.
        """
        ctx1 = {**context, 'mag': mag1}
        ctx2 = {**context, 'mag': mag2}

        a1 = self.association(emb1, **ctx1)
        a2 = self.association(emb2, **ctx2)
        c1 = self.curation.curate_full(a1, emb=emb1, **ctx1)
        c2 = self.curation.curate_full(a2, emb=emb2, **ctx2)

        return {
            'embedding': emb1, 'embedding_aug': emb2,
            'mag1': mag1, 'mag2': mag2,
            'features1': c1['features'], 'features2': c2['features'],
            'cos1': a1['cos_to_anchors'], 'cos2': a2['cos_to_anchors'],
            'tri1': a1['distances'], 'tri2': a2['distances'],
            'nearest': a1['nearest'],
            'assign1': a1['assignment'], 'assign2': a2['assignment'],
            'patchwork1': c1['patchwork'], 'patchwork1_aug': c2['patchwork'],
            'bridge1': c1['bridge'], 'bridge2': c2['bridge'],
        }

    def forward(self, emb, **context):
        return self.observe(emb, **context)