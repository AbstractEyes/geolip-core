"""
GeoLIP Core — Geometric Building Blocks
==========================================
All reusable geometric components. No losses, no training loops.

Components:
  Activations:        SquaredReLU, StarReLU, make_activation
  Anchor Init:        xavier, orthogonal, repulsion
  Constellation:      Triangulation on S^(d-1)
  Patchwork:          Round-robin compartmentalized interpretation
  RelayLayer:         Single constellation relay (vectorized, gated, no attention)
  ConstellationRelay: Per-token geometric layer (O(S), 99.4% at depth 16)
  MagnitudeFlow:      Relay-stack per-compartment magnitude prediction
  AnchorPush:         Push strategies (raw, gru, momentum)
  FlowAttention:      ODE flow in tangent space (historical)

Usage:
    from geolip_core import Constellation, Patchwork, MagnitudeFlow, AnchorPush
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── ACTIVATIONS ──

class SquaredReLU(nn.Module):
    def forward(self, x): return F.relu(x) ** 2

class StarReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.8944)
        self.bias = nn.Parameter(torch.zeros(1) - 0.4472)
    def forward(self, x): return F.relu(x) ** 2 * self.scale + self.bias

ACTIVATIONS = {
    'squared_relu': SquaredReLU, 'star_relu': StarReLU,
    'gelu': lambda: nn.GELU(), 'relu': lambda: nn.ReLU(), 'sigmoid': lambda: nn.Sigmoid(),
}

def make_activation(name='squared_relu'):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


# ── ANCHOR INITIALIZATION ──

def init_anchors_xavier(n, d):
    w = torch.empty(n, d); nn.init.xavier_normal_(w); return F.normalize(w, dim=-1)

def init_anchors_orthogonal(n, d):
    if n <= d:
        Q, _ = torch.linalg.qr(torch.randn(d, n)); return Q.T.contiguous()
    else:
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        return torch.cat([Q.T, F.normalize(torch.randn(n - d, d), dim=-1)], dim=0)

def init_anchors_repulsion(n, d, iters=200, lr=0.05):
    vecs = F.normalize(init_anchors_orthogonal(n, d), dim=-1)
    for _ in range(iters):
        sim = vecs @ vecs.T; sim.fill_diagonal_(-2.0)
        vecs = F.normalize(vecs - lr * vecs[sim.argmax(dim=1)], dim=-1)
    return vecs

INIT_METHODS = {'xavier': init_anchors_xavier, 'orthogonal': init_anchors_orthogonal, 'repulsion': init_anchors_repulsion}


# ── CONSTELLATION ──

class Constellation(nn.Module):
    """Anchors on S^(d-1). Triangulates input embeddings."""
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
            if mask.sum() < 2: mask[:2] = True
            anchors = anchors[mask]
            cos = emb @ anchors.T; tri = 1.0 - cos
            _, nl = cos.max(dim=-1)
            return tri, mask.nonzero(as_tuple=True)[0][nl]
        cos = emb @ anchors.T; tri = 1.0 - cos; _, nearest = cos.max(dim=-1)
        return tri, nearest

    def forward(self, emb, training=False): return self.triangulate(emb, training)


# ── PATCHWORK ──

class Patchwork(nn.Module):
    """Round-robin compartments reading diverse anchor subsets."""
    def __init__(self, n_anchors, n_comp=8, d_comp=64, activation='squared_relu'):
        super().__init__()
        self.n_comp, self.d_comp = n_comp, d_comp
        self.output_dim = n_comp * d_comp
        self.register_buffer('asgn', torch.arange(n_anchors) % n_comp)
        apc = n_anchors // n_comp
        self.comps = nn.ModuleList([
            nn.Sequential(nn.Linear(apc, d_comp*2), make_activation(activation),
                          nn.Linear(d_comp*2, d_comp), nn.LayerNorm(d_comp))
            for _ in range(n_comp)])

    def forward(self, tri):
        return torch.cat([self.comps[k](tri[:, self.asgn == k]) for k in range(self.n_comp)], dim=-1)


# ── RELAY LAYER ──

class RelayLayer(nn.Module):
    """Single constellation relay. Vectorized, gated, no attention.
    Patches → S^(patch_dim-1) → triangulate at 3 SLERP phases → patchwork → gated residual."""
    def __init__(self, input_dim, patch_dim=16, n_anchors=16, n_phases=3, pw_hidden=32, gate_init=-3.0):
        super().__init__()
        assert input_dim % patch_dim == 0
        self.input_dim, self.patch_dim = input_dim, patch_dim
        self.n_patches = input_dim // patch_dim
        self.n_anchors, self.n_phases = n_anchors, n_phases
        P, A, d = self.n_patches, n_anchors, patch_dim

        home = torch.empty(P, A, d); nn.init.xavier_normal_(home.view(P*A, d))
        home = F.normalize(home.view(P, A, d), dim=-1)
        self.register_buffer('home', home)
        self.anchors = nn.Parameter(home.clone())

        tri_dim = n_phases * A
        self.pw_w1 = nn.Parameter(torch.empty(P, tri_dim, pw_hidden))
        self.pw_b1 = nn.Parameter(torch.zeros(1, P, pw_hidden))
        self.pw_w2 = nn.Parameter(torch.empty(P, pw_hidden, d))
        self.pw_b2 = nn.Parameter(torch.zeros(1, P, d))
        for p in range(P):
            nn.init.xavier_normal_(self.pw_w1.data[p])
            nn.init.xavier_normal_(self.pw_w2.data[p])
        self.pw_norm = nn.LayerNorm(d)
        self.gates = nn.Parameter(torch.full((P,), gate_init))
        self.norm = nn.LayerNorm(input_dim)

    def drift(self):
        h, c = F.normalize(self.home, dim=-1), F.normalize(self.anchors, dim=-1)
        return torch.acos((h * c).sum(dim=-1).clamp(-1+1e-7, 1-1e-7))

    def at_phase(self, t):
        h, c = F.normalize(self.home, dim=-1), F.normalize(self.anchors, dim=-1)
        omega = self.drift().unsqueeze(-1); so = omega.sin().clamp(min=1e-7)
        return torch.sin((1-t)*omega)/so * h + torch.sin(t*omega)/so * c

    def forward(self, x):
        B, D = x.shape; P, A, d = self.n_patches, self.n_anchors, self.patch_dim
        patches = self.norm(x).reshape(B, P, d)
        patches_n = F.normalize(patches, dim=-1)
        tris = []
        for t in [0.0, 1/3, 2/3]:
            at = F.normalize(self.at_phase(t), dim=-1)
            tris.append(1.0 - torch.einsum('bpd,pad->bpa', patches_n, at))
        tri = torch.cat(tris, dim=-1)
        h = F.gelu(torch.einsum('bpt,pth->bph', tri, self.pw_w1) + self.pw_b1)
        pw = self.pw_norm(torch.einsum('bph,phd->bpd', h, self.pw_w2) + self.pw_b2)
        gate = self.gates.sigmoid().unsqueeze(0).unsqueeze(-1)
        return x + (gate * pw + (1-gate) * patches).reshape(B, D)


# ── CONSTELLATION RELAY (sequence-aware) ──

class ConstellationRelay(nn.Module):
    """Per-token geometric processing. O(S). Handles (B,D) and (B,S,D)."""
    def __init__(self, dim, n_anchors=16, n_comp=8, d_comp=64,
                 gate_init=-3.0, anchor_init='repulsion', activation='squared_relu'):
        super().__init__()
        self.dim = dim; self.norm = nn.LayerNorm(dim)
        self.constellation = Constellation(n_anchors, dim, anchor_init=anchor_init)
        self.patchwork = Patchwork(n_anchors, n_comp, d_comp, activation)
        self.proj = nn.Linear(self.patchwork.output_dim, dim)
        self.gate = nn.Parameter(torch.full((dim,), gate_init))

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze: x = x.unsqueeze(1)
        B, S, D = x.shape; residual = x
        h_flat = F.normalize(self.norm(x).reshape(B*S, D), dim=-1)
        tri, _ = self.constellation.triangulate(h_flat)
        update = self.proj(self.patchwork(tri)).reshape(B, S, D)
        out = residual + torch.sigmoid(self.gate) * update
        return out.squeeze(1) if squeeze else out


# ── MAGNITUDE FLOW ──

class MagnitudeFlow(nn.Module):
    """Relay-stack per-compartment magnitude. No attention."""
    def __init__(self, dim, n_anchors, hidden_dim=64, n_heads=4,
                 n_layers=2, mag_min=0.1, mag_max=5.0, n_comp=8):
        super().__init__()
        self.dim, self.n_anchors = dim, n_anchors
        self.mag_min, self.mag_max, self.n_comp, self.n_layers = mag_min, mag_max, n_comp, n_layers
        patch_dim = 16; relay_dim = n_comp * patch_dim
        self.patch_dim, self.relay_dim = patch_dim, relay_dim

        self.emb_proj = nn.Linear(dim, relay_dim // 2)
        self.tri_proj = nn.Linear(n_anchors, relay_dim // 4)
        self.ctx_proj = nn.Linear(relay_dim // 2 + relay_dim // 4 + 1, relay_dim)
        self.relays = nn.ModuleList([
            RelayLayer(relay_dim, patch_dim, 16, 3, hidden_dim, -3.0) for _ in range(n_layers)])
        self.mag_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(patch_dim, patch_dim//2), nn.GELU(), nn.Linear(patch_dim//2, 1))
            for _ in range(n_comp)])
        self.register_buffer('stats_bias_cached', torch.zeros(n_comp), persistent=False)

    def update_stats(self, push_diag, anchor_push):
        with torch.no_grad():
            device = self.stats_bias_cached.device
            if anchor_push.strategy == 'momentum' and anchor_push.accumulator is not None:
                mn = anchor_push.accumulator.norm(dim=-1)
                apc = self.n_anchors // self.n_comp
                self.stats_bias_cached = torch.stack([
                    mn[k*apc : (k+1)*apc if k < self.n_comp-1 else self.n_anchors].mean()
                    for k in range(self.n_comp)])
            else: self.stats_bias_cached.zero_()

    def forward(self, emb, triangulation, raw_magnitude):
        B, A = emb.shape[0], self.n_anchors
        x = self.ctx_proj(torch.cat([self.emb_proj(emb), self.tri_proj(triangulation), raw_magnitude], -1))
        for relay in self.relays: x = relay(x)
        patches = x.reshape(B, self.n_comp, self.patch_dim)
        mc = torch.cat([self.mag_heads[k](patches[:, k]) for k in range(self.n_comp)], -1)
        mc = self.mag_min + (self.mag_max - self.mag_min) * torch.sigmoid(mc + self.stats_bias_cached)
        apc = A // self.n_comp
        mag = torch.cat([mc[:, k:k+1].expand(-1, apc if k < self.n_comp-1 else A - k*apc)
                         for k in range(self.n_comp)], -1)
        return mag, mc

    def get_relay_diagnostics(self):
        return [{'layer': i, 'drift_mean': r.drift().mean().item(),
                 'gate_mean': r.gates.sigmoid().mean().item()} for i, r in enumerate(self.relays)]


# ── ANCHOR PUSH ──

def _project_tangent(vec, point):
    return vec - (vec * point).sum(dim=-1, keepdim=True) * point

def _compute_centroids_and_assign(anchors_n, emb_n, label_buffer, device):
    n_a = anchors_n.shape[0]; classes = label_buffer.unique(); n_cls = classes.shape[0]
    centroids = torch.cat([F.normalize(emb_n[label_buffer==c].mean(0, keepdim=True), dim=-1)
                           for c in classes if (label_buffer==c).sum() > 0], dim=0)
    if centroids.shape[0] == 0: return None, None, None, None
    cos = anchors_n @ centroids.T; apc = n_a // n_cls
    assigned = torch.full((n_a,), -1, dtype=torch.long, device=device)
    cc = torch.zeros(n_cls, dtype=torch.long, device=device)
    for idx in cos.flatten().sort(descending=True).indices:
        a, c = (idx // n_cls).item(), (idx % n_cls).item()
        if assigned[a] >= 0 or cc[c] >= apc + 1: continue
        assigned[a] = c; cc[c] += 1
        if (assigned >= 0).all(): break
    u = (assigned < 0).nonzero(as_tuple=True)[0]
    if len(u) > 0: assigned[u] = (anchors_n[u] @ centroids.T).argmax(1)
    nearest = (emb_n @ anchors_n.T).argmax(1)
    util = torch.bincount(nearest, minlength=n_a).float()
    return centroids, assigned, util / util.sum().clamp(min=1), classes

def _perturb_target(target, apc, rank):
    if apc > 1 and rank > 0:
        noise = torch.randn_like(target) * 0.05
        return F.normalize(target + noise - (noise * target).sum() * target, dim=-1)
    return target

class AnchorPush:
    """Configurable anchor push. Strategies: raw, gru, momentum."""
    def __init__(self, strategy, n_anchors, dim, **kw):
        self.strategy, self.n_anchors, self.dim, self.push_count = strategy, n_anchors, dim, 0
        if strategy == 'raw': self.lr = kw.get('lr', 0.1)
        elif strategy == 'momentum':
            self.decay, self.alpha, self.beta = kw.get('decay', 0.9), kw.get('alpha', 0.1), kw.get('beta', 0.05)
            self.util_floor, self.accumulator = kw.get('util_floor', 0.001), None
        elif strategy == 'gru':
            self.ema_decay = kw.get('ema_decay', 0.9); self.z_scale = kw.get('z_scale', 3.0)
            self.r_scale = kw.get('r_scale', 5.0)
            self.prev_pos = self.util_ema = self.drift_ema = None

    @torch.no_grad()
    def push(self, core, emb_buf, lbl_buf):
        anchors = core.constellation.anchors.data; n_a = anchors.shape[0]; device = anchors.device
        emb_n = F.normalize(emb_buf, dim=-1); anchors_n = F.normalize(anchors, dim=-1)
        centroids, assigned, util, classes = _compute_centroids_and_assign(anchors_n, emb_n, lbl_buf, device)
        if centroids is None: return {'moved': 0}
        if hasattr(core, 'anchor_classes'):
            for a in range(n_a): core.anchor_classes[a] = classes[assigned[a]]
        if hasattr(core, 'class_centroids'):
            for i, c in enumerate(classes): core.class_centroids[c] = centroids[i]
        apc = n_a // centroids.shape[0]
        targets = torch.stack([_perturb_target(centroids[assigned[a].item()], apc,
                               (assigned[:a]==assigned[a]).sum().item()) for a in range(n_a)])
        if self.strategy == 'raw':
            for a in range(n_a): anchors[a] = F.normalize(anchors_n[a] + self.lr*(targets[a]-anchors_n[a]), dim=-1)
            d = torch.acos((anchors_n * F.normalize(anchors, dim=-1)).sum(-1).clamp(-1+1e-6, 1-1e-6))
            diag = {'drift_mean': d.mean().item(), 'drift_max': d.max().item()}
        elif self.strategy == 'momentum':
            if self.accumulator is None: self.accumulator = torch.zeros(n_a, self.dim, device=device)
            res = _project_tangent(targets - anchors_n, anchors_n)
            self.accumulator = self.decay * _project_tangent(self.accumulator, anchors_n) + res
            corr = self.alpha * res + self.beta * self.accumulator
            dead = util < self.util_floor
            if dead.any(): corr[dead] = res[dead] * 0.5
            new = F.normalize(anchors_n + corr, dim=-1)
            d = torch.acos((anchors_n * new).sum(-1).clamp(-1+1e-6, 1-1e-6))
            anchors.copy_(new)
            diag = {'drift_mean': d.mean().item(), 'drift_max': d.max().item(),
                    'momentum_mean': self.accumulator.norm(dim=-1).mean().item(), 'dead_count': dead.sum().item()}
        else:
            diag = {}
        diag.update({'moved': n_a, 'n_active': (util > 0).sum().item(),
                     'util_min': util.min().item(), 'util_max': util.max().item()})
        self.push_count += 1; return diag


# ── FLOW ATTENTION (historical) ──

class FlowAttention(nn.Module):
    """3-step Euler flow in tangent space. Superseded by relay."""
    def __init__(self, dim, n_anchors, flow_dim=64, n_steps=3, time_dim=32, gate_init=-3.0):
        super().__init__()
        self.dim, self.flow_dim, self.n_anchors, self.n_steps, self.time_dim = dim, flow_dim, n_anchors, n_steps, time_dim
        self.to_flow = nn.Sequential(nn.Linear(n_anchors+dim, flow_dim), nn.LayerNorm(flow_dim))
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, flow_dim), nn.GELU())
        self.stats_proj = nn.Linear(3, flow_dim, bias=False)
        self.velocity = nn.Sequential(nn.Linear(flow_dim, flow_dim*2), nn.GELU(), nn.Linear(flow_dim*2, flow_dim))
        self.to_correction = nn.Linear(flow_dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((dim,), gate_init))
        self.register_buffer('stats_bias_cached', torch.zeros(flow_dim), persistent=False)

    def update_stats(self, push_diag, anchor_push):
        with torch.no_grad():
            dev = self.stats_proj.weight.device
            mn = anchor_push.accumulator.norm(dim=-1) if (anchor_push.strategy=='momentum' and anchor_push.accumulator is not None) else torch.zeros(self.n_anchors, device=dev)
            dr = torch.tensor(push_diag.get('drift_mean',0.0), device=dev).expand(self.n_anchors)
            ut = torch.tensor(push_diag.get('util_max',0.0), device=dev).expand(self.n_anchors)
            self.stats_bias_cached = self.stats_proj(torch.stack([mn, ut, dr], -1)).mean(0)

    def forward(self, emb, constellation):
        B, D, dev = *emb.shape, emb.device
        tri = emb @ F.normalize(constellation.anchors, dim=-1).T
        z = self.to_flow(torch.cat([tri, emb], -1)); dt = 1.0/self.n_steps
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=dev) / half)
        for s in range(self.n_steps):
            args = (s*dt)*freqs; t_emb = torch.cat([args.sin(), args.cos()])
            z = z + dt * (self.velocity(z + self.time_mlp(t_emb)) + self.stats_bias_cached)
        c = self.to_correction(z); c = c - (c*emb).sum(-1,keepdim=True)*emb
        return F.normalize(emb + torch.sigmoid(self.gate)*c, dim=-1)


# ── GEOMETRIC AUTOGRAD ──

class GeometricAutograd(torch.autograd.Function):
    """Manifold-aware gradient correction on S^(D-1). Forward: identity."""
    @staticmethod
    def forward(ctx, emb, anchors, tang_strength, sep_strength):
        ctx.save_for_backward(emb, anchors); ctx.tang, ctx.sep = tang_strength, sep_strength
        return emb

    @staticmethod
    def backward(ctx, grad):
        emb, anchors = ctx.saved_tensors
        dot = (grad * emb).sum(-1, keepdim=True)
        corrected = grad - ctx.tang * dot * emb
        if ctx.sep > 0:
            an = F.normalize(anchors.detach(), dim=-1)
            nearest = an[(emb @ an.T).argmax(-1)]
            toward = (corrected * nearest).sum(-1, keepdim=True)
            corrected = corrected - ctx.sep * F.relu(toward) * nearest
        return corrected, None, None, None


# ── UTILITIES ──

def param_count(module, name=""):
    t = sum(p.numel() for p in module.parameters())
    tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if name: print(f"  {name}: {t:,} ({tr:,} trainable)")
    return t, tr

def model_summary(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total:,}")
    for n, m in model.named_children():
        c = sum(p.numel() for p in m.parameters())
        if c > 0: print(f"    {n}: {c:,}")
    return total