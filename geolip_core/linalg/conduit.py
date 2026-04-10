"""
FLEighConduit — Evidence-Emitting Eigendecomposition
======================================================
Extends FLEigh with judicial conduit telemetry.
Read-only observation of the solver's internal states.

Council-ratified specification (11 rounds, 3 AI participants):
  - Theorem 1: Lens Preservation (shared arithmetic path)
  - Theorem 2: Dynamic Non-Reconstructibility (friction, settle, order)
  - Theorem 4: Continuity (static continuous, dynamic piecewise)
  - Theorem 5: Gauge-Safe Directional Observation (sign canonicalization)

Classes:
  ConduitPacket  — fixed-shape tensor bundle, batch-first, dimension-agnostic
  FLEighConduit  — extends FLEigh, emits ConduitPacket

Usage:
    from geolip_core.linalg.conduit import FLEighConduit

    solver = FLEighConduit()
    packet = solver(A)          # A: (B, n, n) symmetric
    packet.eigenvalues          # (B, n)
    packet.friction             # (B, n) — per-root solver struggle
    packet.settle               # (B, n) — iterations to convergence
    packet.extraction_order     # (B, n) — root extraction sequence

    # Standard eigenpairs (identical to FLEigh)
    evals, evecs = packet.eigenpairs()

License: MIT

Author: AbstractPhil + Claude 4.6 Opus Extended
Assistants: Gemini Pro, GPT 5.4 Extended Thinking
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConduitPacket:
    """Fixed-shape telemetry from FLEighConduit.

    All tensors batch-first, dimension-agnostic.
    Production packet: scalar-dominant, bounded overhead.
    Research fields (Mstore, trajectories) populated only when requested.
    """

    # ── Spectral evidence (static, deterministic) ──
    eigenvalues: Tensor          # (B, n) sorted ascending
    eigenvectors: Tensor         # (B, n, n) sign-canonicalized
    char_coeffs: Tensor          # (B, n) elementary symmetric polys, monic 1 omitted

    # ── Adjudication evidence (dynamic, non-reconstructible) ──
    friction: Tensor             # (B, n) per-root Σ 1/(|p'(z_t)| + δ)
    settle: Tensor               # (B, n) iterations to convergence per root
    extraction_order: Tensor     # (B, n) which root found first (0-indexed)
    refinement_residual: Tensor  # (B,) ||V^T V - I||_F after Newton-Schulz

    # ── Release fidelity ──
    # Note: release_residual is owned by SVDConduit layer in full architecture.
    # Included here for v1 convenience when enc_out matrix M is provided.
    release_residual: Optional[Tensor] = None  # (B,) ||M - U diag(S) Vt||²

    # ── Research mode (populated only with research=True) ──
    mstore: Optional[Tensor] = None     # (n+1, B, n, n) FL matrix states
    z_trajectory: Optional[Tensor] = None  # (B, n, laguerre_iters) root guesses
    dp_trajectory: Optional[Tensor] = None # (B, n, laguerre_iters) p' at each step

    def eigenpairs(self) -> Tuple[Tensor, Tensor]:
        """Standard output matching FLEigh contract."""
        return self.eigenvalues, self.eigenvectors


def canonicalize_eigenvectors(V: Tensor) -> Tensor:
    """Force deterministic sign convention on eigenvector columns.

    For each column (eigenvector), flip sign so the entry with
    largest absolute value is positive. Resolves the gauge ambiguity
    that otherwise causes identical matrices to produce different
    embeddings on S^(n²-1).

    Args:
        V: (B, n, n) eigenvector matrix (columns are eigenvectors)
    Returns:
        V with deterministic signs
    """
    # Find index of max absolute value per column
    max_idx = V.abs().argmax(dim=-2, keepdim=True)  # (B, 1, n)
    sign = V.gather(-2, max_idx).sign()               # (B, 1, n)
    return V * sign


class FLEighConduit(nn.Module):
    """Evidence-emitting eigendecomposition.

    Identical arithmetic to FLEigh. Captures telemetry at phase
    boundaries without altering the numerical path.

    Phases (shared with FLEigh):
      1. FL characteristic polynomial (fp64, n bmm)
      2. Laguerre root-finding + Newton polish (with telemetry capture)
      3. FL adjugate eigenvectors (fp64 Horner + max-col)
      4. Newton-Schulz orthogonalization (fp32, 2 iters)
      5. Rayleigh quotient refinement (fp32, 2 bmm)

    Args:
        laguerre_iters: Root-finding iterations per eigenvalue (default 5)
        polish_iters: Newton refinement iterations (default 3)
        ns_iters: Newton-Schulz orthogonalization iterations (default 2)
        friction_delta: stability constant for friction computation (default 1e-8)
        settle_threshold: convergence threshold for settle count (default 1e-6)
        research: if True, populate full trajectory and Mstore fields
    """

    def __init__(self, laguerre_iters: int = 5, polish_iters: int = 3,
                 ns_iters: int = 2, friction_delta: float = 1e-8,
                 settle_threshold: float = 1e-6, research: bool = False):
        super().__init__()
        self.laguerre_iters = laguerre_iters
        self.polish_iters = polish_iters
        self.ns_iters = ns_iters
        self.friction_delta = friction_delta
        self.settle_threshold = settle_threshold
        self.research = research

    def forward(self, A: Tensor) -> ConduitPacket:
        """Evidence-emitting eigendecomposition.

        Args:
            A: (B, n, n) symmetric matrix batch

        Returns:
            ConduitPacket with eigenpairs + judicial telemetry
        """
        B, n, _ = A.shape
        device = A.device

        # ════════════════════════════════════════════
        # Phase 1: Faddeev-LeVerrier (fp64)
        # ════════════════════════════════════════════
        scale = (torch.linalg.norm(A.reshape(B, -1), dim=-1) / math.sqrt(n)).clamp(min=1e-12)
        As = A / scale[:, None, None]

        Ad = As.double()
        eye_d = torch.eye(n, device=device, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1)
        c = torch.zeros(B, n + 1, device=device, dtype=torch.float64)
        c[:, n] = 1.0
        Mstore = torch.zeros(n + 1, B, n, n, device=device, dtype=torch.float64)
        Mk = torch.zeros(B, n, n, device=device, dtype=torch.float64)
        for k in range(1, n + 1):
            Mk = torch.bmm(Ad, Mk) + c[:, n - k + 1, None, None] * eye_d
            Mstore[k] = Mk
            c[:, n - k] = -(Ad * Mk).sum((-2, -1)) / k

        # Capture characteristic coefficients (omit monic leading 1)
        char_coeffs = c[:, :n].float()  # (B, n)

        # ════════════════════════════════════════════
        # Phase 2: Laguerre + deflation + Newton polish
        #          WITH TELEMETRY CAPTURE
        # ════════════════════════════════════════════
        use_f64 = n > 6
        dt = torch.float64 if use_f64 else torch.float32
        cl = c.to(dt).clone().detach()
        roots = torch.zeros(B, n, device=device, dtype=dt)
        zi = As.to(dt).diagonal(dim1=-2, dim2=-1).sort(dim=-1).values.detach()
        zi = zi + torch.linspace(-1e-4, 1e-4, n, device=device, dtype=dt).unsqueeze(0)

        # Telemetry buffers
        friction = torch.zeros(B, n, device=device, dtype=torch.float32)
        settle = torch.full((B, n), float(self.laguerre_iters),
                           device=device, dtype=torch.float32)
        extraction_order = torch.zeros(B, n, device=device, dtype=torch.float32)

        # Research buffers
        if self.research:
            z_traj = torch.zeros(B, n, self.laguerre_iters,
                                device=device, dtype=torch.float32)
            dp_traj = torch.zeros(B, n, self.laguerre_iters,
                                 device=device, dtype=torch.float32)

        for ri in range(n):
            deg = n - ri
            z = zi[:, ri]

            for lag_iter in range(self.laguerre_iters):
                # Horner evaluation
                pv = cl[:, deg]
                dp = torch.zeros(B, device=device, dtype=dt)
                d2 = torch.zeros(B, device=device, dtype=dt)
                for j in range(deg - 1, -1, -1):
                    d2 = d2 * z + dp
                    dp = dp * z + pv
                    pv = pv * z + cl[:, j]

                # ── Telemetry capture ──
                dp_abs = dp.abs().float()
                friction[:, ri] += 1.0 / (dp_abs + self.friction_delta)

                # Settle detection
                pv_abs = pv.abs().float()
                just_settled = (pv_abs < self.settle_threshold) & \
                               (settle[:, ri] == float(self.laguerre_iters))
                settle[:, ri] = torch.where(just_settled,
                                           torch.full_like(settle[:, ri], float(lag_iter)),
                                           settle[:, ri])

                if self.research:
                    z_traj[:, ri, lag_iter] = z.float()
                    dp_traj[:, ri, lag_iter] = dp_abs

                # Laguerre step (unchanged arithmetic)
                ok = pv.abs() > 1e-30
                ps = torch.where(ok, pv, torch.ones_like(pv))
                G = torch.where(ok, dp / ps, torch.zeros_like(dp))
                H = G * G - torch.where(ok, 2.0 * d2 / ps, torch.zeros_like(d2))
                disc = ((deg - 1.0) * (deg * H - G * G)).clamp(min=0.0)
                sq = torch.sqrt(disc)
                gp = G + sq
                gm = G - sq
                den = torch.where(gp.abs() >= gm.abs(), gp, gm)
                dok = den.abs() > 1e-20
                ds = torch.where(dok, den, torch.ones_like(den))
                z = z - torch.where(dok, float(deg) / ds, torch.zeros_like(den))

            roots[:, ri] = z
            extraction_order[:, ri] = float(ri)

            # Synthetic division (deflation)
            b = cl[:, deg]
            for j in range(deg - 1, 0, -1):
                bn = cl[:, j] + z * b
                cl[:, j] = b
                b = bn
            cl[:, 0] = b

        # Newton polish on original polynomial
        roots = roots.double()
        for _ in range(self.polish_iters):
            pv = torch.ones(B, n, device=device, dtype=torch.float64)
            dp = torch.zeros(B, n, device=device, dtype=torch.float64)
            for j in range(n - 1, -1, -1):
                dp = dp * roots + pv
                pv = pv * roots + c[:, j:j + 1]
            ok = dp.abs() > 1e-30
            dps = torch.where(ok, dp, torch.ones_like(dp))
            roots = roots - torch.where(ok, pv / dps, torch.zeros_like(pv))

        # ════════════════════════════════════════════
        # Phase 3: FL adjugate eigenvectors (fp64)
        # ════════════════════════════════════════════
        lam = roots
        R = Mstore[1].unsqueeze(1).expand(-1, n, -1, -1).clone()
        for k in range(2, n + 1):
            R = R * lam[:, :, None, None] + Mstore[k].unsqueeze(1)
        cnorms = R.norm(dim=-2)
        best = cnorms.argmax(dim=-1)
        idx = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, 1)
        vec = R.gather(-1, idx).squeeze(-1)
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-30)
        V = vec.float().transpose(-2, -1)

        # ════════════════════════════════════════════
        # Phase 4: Newton-Schulz orthogonalization
        # ════════════════════════════════════════════
        eye_f = torch.eye(n, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        Y = torch.bmm(V.transpose(-2, -1), V)
        X = eye_f.clone()
        for _ in range(self.ns_iters):
            T = 3.0 * eye_f - Y
            X = 0.5 * torch.bmm(X, T)
            Y = 0.5 * torch.bmm(T, Y)
        V = torch.bmm(V, X)

        # Telemetry: orthogonality residual after NS
        VtV = torch.bmm(V.transpose(-2, -1), V)
        refinement_residual = (VtV - eye_f).pow(2).sum((-2, -1)).sqrt()  # (B,)

        # ════════════════════════════════════════════
        # Phase 5: Rayleigh quotient refinement
        # ════════════════════════════════════════════
        AV = torch.bmm(A, V)
        evals = (V * AV).sum(dim=-2)

        se, perm = evals.sort(dim=-1)
        sv = V.gather(-1, perm.unsqueeze(-2).expand_as(V))

        # Reorder telemetry to match sorted eigenvalue order
        friction_sorted = friction.gather(-1, perm)
        settle_sorted = settle.gather(-1, perm)
        # extraction_order stays as-is — it records the ORIGINAL extraction sequence

        # ════════════════════════════════════════════
        # Gauge canonicalization
        # ════════════════════════════════════════════
        sv = canonicalize_eigenvectors(sv)

        # ════════════════════════════════════════════
        # Build packet
        # ════════════════════════════════════════════
        packet = ConduitPacket(
            eigenvalues=se,
            eigenvectors=sv,
            char_coeffs=char_coeffs,
            friction=friction_sorted,
            settle=settle_sorted,
            extraction_order=extraction_order,
            refinement_residual=refinement_residual,
        )

        if self.research:
            packet.mstore = Mstore
            packet.z_trajectory = z_traj
            packet.dp_trajectory = dp_traj

        return packet


# ── Regression parity test ──

def verify_parity(A: Tensor, atol: float = 1e-5) -> bool:
    """Verify FLEighConduit produces identical eigenpairs to FLEigh.

    Args:
        A: (B, n, n) symmetric test matrices
        atol: absolute tolerance

    Returns:
        True if eigenpairs match within tolerance
    """
    from geolip_core.linalg.eigh import FLEigh

    ref_evals, ref_evecs = FLEigh()(A)
    packet = FLEighConduit()(A)
    cond_evals, cond_evecs = packet.eigenpairs()

    evals_match = torch.allclose(ref_evals, cond_evals, atol=atol)

    # Eigenvectors may differ by sign — compare via absolute inner products
    dots = (ref_evecs * cond_evecs).sum(dim=-2).abs()
    evecs_match = (dots > 1.0 - atol).all()

    return evals_match and evecs_match