"""
kernel.py — Generalized batched thin SVD + Procrustes alignment.

Part of the GEOLIP ecosystem.
Repository: AbstractEyes/geolip-core
Package: geolip

Provides:
  batched_svd(A)                    — Auto-dispatched thin SVD for (B, M, N)
  batched_svd2(A)                   — Fused Triton kernel for N=2
  batched_svd3(A)                   — Fused Triton kernel for N=3
  gram_eigh_svd(A)                  — Gram + eigh hybrid for any N
  newton_schulz_invsqrt(G)          — Batched G^{-1/2} via pure bmm
  batched_procrustes(src, tgt)      — Subspace-preserving Procrustes alignment

Performance (NVIDIA RTX PRO 6000 Blackwell, B=512, M=1024):
  N=2:   0.021ms  (3,850× vs torch)
  N=3:   0.022ms  (5,488× vs torch)
  N=8:   0.290ms  (584× vs torch)
  N=32:  0.781ms  (388× vs torch)

Mathematical lineage:
  Eckart-Young (1936), Jacobi (1846), Golub-Reinsch (1970), Batcher (1968)

Author: AbstractPhil + Claude Opus 4.6
License: Apache 2.0
"""

import math
import torch
import torch.nn.functional as F

__all__ = [
    'batched_svd',
    'batched_svd2',
    'batched_svd3',
    'batched_svd4',
    'batched_svd5',
    'batched_svd6',
    'gram_eigh_svd',
    'newton_schulz_invsqrt',
    'batched_procrustes',
    'HAS_TRITON',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRITON FUSED KERNELS (N=2, N=3)
# ═══════════════════════════════════════════════════════════════════════════════

HAS_TRITON = False

try:
    import triton
    import triton.language as tl

    # ── N=2: Closed-form Jacobi rotation ─────────────────────────────────

    @triton.jit
    def _svd2_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        DTYPE: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        base = bid * M * 2
        g00 = tl.zeros([], dtype=DTYPE)
        g01 = tl.zeros([], dtype=DTYPE)
        g11 = tl.zeros([], dtype=DTYPE)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M)
            row_idx = block_start + offs
            mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 2 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 2 + 1, mask=mask, other=0.0).to(DTYPE)
            g00 += tl.sum(a0 * a0)
            g01 += tl.sum(a0 * a1)
            g11 += tl.sum(a1 * a1)
        # Jacobi rotation (single step, no iteration needed for 2×2)
        off_diag = g01
        diag_diff = g11 - g00
        abs_off = tl.abs(off_diag)
        tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
        t = tl.where(abs_off > EPS,
            tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)),
            0.0)
        c = 1.0 / tl.sqrt(1.0 + t * t)
        s = t * c
        eig0 = c * c * g00 - 2.0 * s * c * g01 + s * s * g11
        eig1 = s * s * g00 + 2.0 * s * c * g01 + c * c * g11
        s0 = tl.sqrt(tl.maximum(eig0, EPS))
        s1 = tl.sqrt(tl.maximum(eig1, EPS))
        v00 = c; v01 = s; v10 = -s; v11 = c
        # Sort descending
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00; v00 = tl.where(do_swap, v01, v00); v01 = tl.where(do_swap, tv, v01)
        tv = v10; v10 = tl.where(do_swap, v11, v10); v11 = tl.where(do_swap, tv, v11)
        # Write S, Vh
        tl.store(S_ptr + bid * 2 + 0, s0)
        tl.store(S_ptr + bid * 2 + 1, s1)
        vh_base = bid * 4
        tl.store(Vh_ptr + vh_base + 0, v00)
        tl.store(Vh_ptr + vh_base + 1, v10)
        tl.store(Vh_ptr + vh_base + 2, v01)
        tl.store(Vh_ptr + vh_base + 3, v11)
        # U recovery
        inv_s0 = 1.0 / (s0 + EPS)
        inv_s1 = 1.0 / (s1 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M)
            row_idx = block_start + offs
            mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 2 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 2 + 1, mask=mask, other=0.0).to(DTYPE)
            u0 = (a0 * v00 + a1 * v10) * inv_s0
            u1 = (a0 * v01 + a1 * v11) * inv_s1
            u_base = bid * M * 2
            tl.store(U_ptr + u_base + row_idx * 2 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 2 + 1, u1, mask=mask)

    # ── N=3: Cyclic Jacobi in scalar registers ───────────────────────────

    @triton.jit
    def _svd3_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        JACOBI_ITERS: tl.constexpr,
        DTYPE: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        g00 = tl.zeros([], dtype=DTYPE); g01 = tl.zeros([], dtype=DTYPE)
        g02 = tl.zeros([], dtype=DTYPE); g11 = tl.zeros([], dtype=DTYPE)
        g12 = tl.zeros([], dtype=DTYPE); g22 = tl.zeros([], dtype=DTYPE)
        base = bid * M * 3
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 3 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 3 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 3 + 2, mask=mask, other=0.0).to(DTYPE)
            g00 += tl.sum(a0 * a0); g01 += tl.sum(a0 * a1); g02 += tl.sum(a0 * a2)
            g11 += tl.sum(a1 * a1); g12 += tl.sum(a1 * a2); g22 += tl.sum(a2 * a2)
        v00 = tl.full([], 1.0, dtype=DTYPE); v01 = tl.zeros([], dtype=DTYPE); v02 = tl.zeros([], dtype=DTYPE)
        v10 = tl.zeros([], dtype=DTYPE); v11 = tl.full([], 1.0, dtype=DTYPE); v12 = tl.zeros([], dtype=DTYPE)
        v20 = tl.zeros([], dtype=DTYPE); v21 = tl.zeros([], dtype=DTYPE); v22 = tl.full([], 1.0, dtype=DTYPE)
        for _ in range(JACOBI_ITERS):
            # pair (0,1)
            off_diag = g01; diag_diff = g11 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g01 + s*s*g11; ng11 = s*s*g00 + 2.0*s*c*g01 + c*c*g11
            ng02 = c*g02 - s*g12; ng12 = s*g02 + c*g12
            g00 = ng00; g11 = ng11; g01 = tl.zeros([], dtype=DTYPE); g02 = ng02; g12 = ng12
            nv00 = c*v00 - s*v01; nv01 = s*v00 + c*v01
            nv10 = c*v10 - s*v11; nv11 = s*v10 + c*v11
            nv20 = c*v20 - s*v21; nv21 = s*v20 + c*v21
            v00 = nv00; v01 = nv01; v10 = nv10; v11 = nv11; v20 = nv20; v21 = nv21
            # pair (0,2)
            off_diag = g02; diag_diff = g22 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g02 + s*s*g22; ng22 = s*s*g00 + 2.0*s*c*g02 + c*c*g22
            ng01 = c*g01 - s*g12; ng12b = s*g01 + c*g12
            g00 = ng00; g22 = ng22; g02 = tl.zeros([], dtype=DTYPE); g01 = ng01; g12 = ng12b
            nv00 = c*v00 - s*v02; nv02 = s*v00 + c*v02
            nv10 = c*v10 - s*v12; nv12 = s*v10 + c*v12
            nv20 = c*v20 - s*v22; nv22 = s*v20 + c*v22
            v00 = nv00; v02 = nv02; v10 = nv10; v12 = nv12; v20 = nv20; v22 = nv22
            # pair (1,2)
            off_diag = g12; diag_diff = g22 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g12 + s*s*g22; ng22 = s*s*g11 + 2.0*s*c*g12 + c*c*g22
            ng01 = c*g01 - s*g02; ng02b = s*g01 + c*g02
            g11 = ng11; g22 = ng22; g12 = tl.zeros([], dtype=DTYPE); g01 = ng01; g02 = ng02b
            nv01 = c*v01 - s*v02; nv02 = s*v01 + c*v02
            nv11 = c*v11 - s*v12; nv12 = s*v11 + c*v12
            nv21 = c*v21 - s*v22; nv22 = s*v21 + c*v22
            v01 = nv01; v02 = nv02; v11 = nv11; v12 = nv12; v21 = nv21; v22 = nv22
        # Sort descending
        s0 = tl.sqrt(tl.maximum(g00, EPS))
        s1 = tl.sqrt(tl.maximum(g11, EPS))
        s2 = tl.sqrt(tl.maximum(g22, EPS))
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00; v00 = tl.where(do_swap, v01, v00); v01 = tl.where(do_swap, tv, v01)
        tv = v10; v10 = tl.where(do_swap, v11, v10); v11 = tl.where(do_swap, tv, v11)
        tv = v20; v20 = tl.where(do_swap, v21, v20); v21 = tl.where(do_swap, tv, v21)
        do_swap = s0 < s2
        s0, s2 = tl.where(do_swap, s2, s0), tl.where(do_swap, s0, s2)
        tv = v00; v00 = tl.where(do_swap, v02, v00); v02 = tl.where(do_swap, tv, v02)
        tv = v10; v10 = tl.where(do_swap, v12, v10); v12 = tl.where(do_swap, tv, v12)
        tv = v20; v20 = tl.where(do_swap, v22, v20); v22 = tl.where(do_swap, tv, v22)
        do_swap = s1 < s2
        s1, s2 = tl.where(do_swap, s2, s1), tl.where(do_swap, s1, s2)
        tv = v01; v01 = tl.where(do_swap, v02, v01); v02 = tl.where(do_swap, tv, v02)
        tv = v11; v11 = tl.where(do_swap, v12, v11); v12 = tl.where(do_swap, tv, v12)
        tv = v21; v21 = tl.where(do_swap, v22, v21); v22 = tl.where(do_swap, tv, v22)
        # Write S
        s_base = bid * 3
        tl.store(S_ptr + s_base + 0, s0)
        tl.store(S_ptr + s_base + 1, s1)
        tl.store(S_ptr + s_base + 2, s2)
        # Write Vh = V^T
        vh_base = bid * 9
        tl.store(Vh_ptr + vh_base + 0, v00); tl.store(Vh_ptr + vh_base + 1, v10); tl.store(Vh_ptr + vh_base + 2, v20)
        tl.store(Vh_ptr + vh_base + 3, v01); tl.store(Vh_ptr + vh_base + 4, v11); tl.store(Vh_ptr + vh_base + 5, v21)
        tl.store(Vh_ptr + vh_base + 6, v02); tl.store(Vh_ptr + vh_base + 7, v12); tl.store(Vh_ptr + vh_base + 8, v22)
        # U recovery
        inv_s0 = 1.0 / (s0 + EPS); inv_s1 = 1.0 / (s1 + EPS); inv_s2 = 1.0 / (s2 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 3 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 3 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 3 + 2, mask=mask, other=0.0).to(DTYPE)
            u0 = (a0 * v00 + a1 * v10 + a2 * v20) * inv_s0
            u1 = (a0 * v01 + a1 * v11 + a2 * v21) * inv_s1
            u2 = (a0 * v02 + a1 * v12 + a2 * v22) * inv_s2
            u_base = bid * M * 3
            tl.store(U_ptr + u_base + row_idx * 3 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 3 + 1, u1, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 3 + 2, u2, mask=mask)

    # ── N=4: Cyclic Jacobi (6 pair sweep) ────────────────────────────────

    @triton.jit
    def _svd4_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        JACOBI_ITERS: tl.constexpr,
        DTYPE: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        base = bid * M * 4
        g00 = tl.zeros([], dtype=DTYPE); g01 = tl.zeros([], dtype=DTYPE)
        g02 = tl.zeros([], dtype=DTYPE); g03 = tl.zeros([], dtype=DTYPE)
        g11 = tl.zeros([], dtype=DTYPE); g12 = tl.zeros([], dtype=DTYPE)
        g13 = tl.zeros([], dtype=DTYPE); g22 = tl.zeros([], dtype=DTYPE)
        g23 = tl.zeros([], dtype=DTYPE); g33 = tl.zeros([], dtype=DTYPE)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 4 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 4 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 4 + 2, mask=mask, other=0.0).to(DTYPE)
            a3 = tl.load(A_ptr + base + row_idx * 4 + 3, mask=mask, other=0.0).to(DTYPE)
            g00 += tl.sum(a0 * a0); g01 += tl.sum(a0 * a1); g02 += tl.sum(a0 * a2); g03 += tl.sum(a0 * a3)
            g11 += tl.sum(a1 * a1); g12 += tl.sum(a1 * a2); g13 += tl.sum(a1 * a3)
            g22 += tl.sum(a2 * a2); g23 += tl.sum(a2 * a3); g33 += tl.sum(a3 * a3)
        v00 = tl.full([], 1.0, dtype=DTYPE); v01 = tl.zeros([], dtype=DTYPE); v02 = tl.zeros([], dtype=DTYPE); v03 = tl.zeros([], dtype=DTYPE)
        v10 = tl.zeros([], dtype=DTYPE); v11 = tl.full([], 1.0, dtype=DTYPE); v12 = tl.zeros([], dtype=DTYPE); v13 = tl.zeros([], dtype=DTYPE)
        v20 = tl.zeros([], dtype=DTYPE); v21 = tl.zeros([], dtype=DTYPE); v22 = tl.full([], 1.0, dtype=DTYPE); v23 = tl.zeros([], dtype=DTYPE)
        v30 = tl.zeros([], dtype=DTYPE); v31 = tl.zeros([], dtype=DTYPE); v32 = tl.zeros([], dtype=DTYPE); v33 = tl.full([], 1.0, dtype=DTYPE)
        for _ in range(JACOBI_ITERS):
            # pair (0,1)
            off_diag = g01; diag_diff = g11 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g01 + s*s*g11; ng11 = s*s*g00 + 2.0*s*c*g01 + c*c*g11
            ng02 = c*g02 - s*g12; ng12 = s*g02 + c*g12
            ng03 = c*g03 - s*g13; ng13 = s*g03 + c*g13
            g00 = ng00; g11 = ng11; g01 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g12 = ng12; g03 = ng03; g13 = ng13
            nv00 = c*v00 - s*v01; nv01 = s*v00 + c*v01
            nv10 = c*v10 - s*v11; nv11 = s*v10 + c*v11
            nv20 = c*v20 - s*v21; nv21 = s*v20 + c*v21
            nv30 = c*v30 - s*v31; nv31 = s*v30 + c*v31
            v00 = nv00; v01 = nv01; v10 = nv10; v11 = nv11
            v20 = nv20; v21 = nv21; v30 = nv30; v31 = nv31
            # pair (0,2)
            off_diag = g02; diag_diff = g22 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g02 + s*s*g22; ng22 = s*s*g00 + 2.0*s*c*g02 + c*c*g22
            ng01 = c*g01 - s*g12; ng12 = s*g01 + c*g12
            ng03 = c*g03 - s*g23; ng23 = s*g03 + c*g23
            g00 = ng00; g22 = ng22; g02 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g12 = ng12; g03 = ng03; g23 = ng23
            nv00 = c*v00 - s*v02; nv02 = s*v00 + c*v02
            nv10 = c*v10 - s*v12; nv12 = s*v10 + c*v12
            nv20 = c*v20 - s*v22; nv22 = s*v20 + c*v22
            nv30 = c*v30 - s*v32; nv32 = s*v30 + c*v32
            v00 = nv00; v02 = nv02; v10 = nv10; v12 = nv12
            v20 = nv20; v22 = nv22; v30 = nv30; v32 = nv32
            # pair (0,3)
            off_diag = g03; diag_diff = g33 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g03 + s*s*g33; ng33 = s*s*g00 + 2.0*s*c*g03 + c*c*g33
            ng01 = c*g01 - s*g13; ng13 = s*g01 + c*g13
            ng02 = c*g02 - s*g23; ng23 = s*g02 + c*g23
            g00 = ng00; g33 = ng33; g03 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g13 = ng13; g02 = ng02; g23 = ng23
            nv00 = c*v00 - s*v03; nv03 = s*v00 + c*v03
            nv10 = c*v10 - s*v13; nv13 = s*v10 + c*v13
            nv20 = c*v20 - s*v23; nv23 = s*v20 + c*v23
            nv30 = c*v30 - s*v33; nv33 = s*v30 + c*v33
            v00 = nv00; v03 = nv03; v10 = nv10; v13 = nv13
            v20 = nv20; v23 = nv23; v30 = nv30; v33 = nv33
            # pair (1,2)
            off_diag = g12; diag_diff = g22 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g12 + s*s*g22; ng22 = s*s*g11 + 2.0*s*c*g12 + c*c*g22
            ng01 = c*g01 - s*g02; ng02 = s*g01 + c*g02
            ng13 = c*g13 - s*g23; ng23 = s*g13 + c*g23
            g11 = ng11; g22 = ng22; g12 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g02 = ng02; g13 = ng13; g23 = ng23
            nv01 = c*v01 - s*v02; nv02 = s*v01 + c*v02
            nv11 = c*v11 - s*v12; nv12 = s*v11 + c*v12
            nv21 = c*v21 - s*v22; nv22 = s*v21 + c*v22
            nv31 = c*v31 - s*v32; nv32 = s*v31 + c*v32
            v01 = nv01; v02 = nv02; v11 = nv11; v12 = nv12
            v21 = nv21; v22 = nv22; v31 = nv31; v32 = nv32
            # pair (1,3)
            off_diag = g13; diag_diff = g33 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g13 + s*s*g33; ng33 = s*s*g11 + 2.0*s*c*g13 + c*c*g33
            ng01 = c*g01 - s*g03; ng03 = s*g01 + c*g03
            ng12 = c*g12 - s*g23; ng23 = s*g12 + c*g23
            g11 = ng11; g33 = ng33; g13 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g03 = ng03; g12 = ng12; g23 = ng23
            nv01 = c*v01 - s*v03; nv03 = s*v01 + c*v03
            nv11 = c*v11 - s*v13; nv13 = s*v11 + c*v13
            nv21 = c*v21 - s*v23; nv23 = s*v21 + c*v23
            nv31 = c*v31 - s*v33; nv33 = s*v31 + c*v33
            v01 = nv01; v03 = nv03; v11 = nv11; v13 = nv13
            v21 = nv21; v23 = nv23; v31 = nv31; v33 = nv33
            # pair (2,3)
            off_diag = g23; diag_diff = g33 - g22; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng22 = c*c*g22 - 2.0*s*c*g23 + s*s*g33; ng33 = s*s*g22 + 2.0*s*c*g23 + c*c*g33
            ng02 = c*g02 - s*g03; ng03 = s*g02 + c*g03
            ng12 = c*g12 - s*g13; ng13 = s*g12 + c*g13
            g22 = ng22; g33 = ng33; g23 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g03 = ng03; g12 = ng12; g13 = ng13
            nv02 = c*v02 - s*v03; nv03 = s*v02 + c*v03
            nv12 = c*v12 - s*v13; nv13 = s*v12 + c*v13
            nv22 = c*v22 - s*v23; nv23 = s*v22 + c*v23
            nv32 = c*v32 - s*v33; nv33 = s*v32 + c*v33
            v02 = nv02; v03 = nv03; v12 = nv12; v13 = nv13
            v22 = nv22; v23 = nv23; v32 = nv32; v33 = nv33
        s0 = tl.sqrt(tl.maximum(g00, EPS))
        s1 = tl.sqrt(tl.maximum(g11, EPS))
        s2 = tl.sqrt(tl.maximum(g22, EPS))
        s3 = tl.sqrt(tl.maximum(g33, EPS))
        # Sort descending (selection sort; swap columns of V when swapping singular values)
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00; v00 = tl.where(do_swap, v01, v00); v01 = tl.where(do_swap, tv, v01)
        tv = v10; v10 = tl.where(do_swap, v11, v10); v11 = tl.where(do_swap, tv, v11)
        tv = v20; v20 = tl.where(do_swap, v21, v20); v21 = tl.where(do_swap, tv, v21)
        tv = v30; v30 = tl.where(do_swap, v31, v30); v31 = tl.where(do_swap, tv, v31)
        do_swap = s0 < s2
        s0, s2 = tl.where(do_swap, s2, s0), tl.where(do_swap, s0, s2)
        tv = v00; v00 = tl.where(do_swap, v02, v00); v02 = tl.where(do_swap, tv, v02)
        tv = v10; v10 = tl.where(do_swap, v12, v10); v12 = tl.where(do_swap, tv, v12)
        tv = v20; v20 = tl.where(do_swap, v22, v20); v22 = tl.where(do_swap, tv, v22)
        tv = v30; v30 = tl.where(do_swap, v32, v30); v32 = tl.where(do_swap, tv, v32)
        do_swap = s0 < s3
        s0, s3 = tl.where(do_swap, s3, s0), tl.where(do_swap, s0, s3)
        tv = v00; v00 = tl.where(do_swap, v03, v00); v03 = tl.where(do_swap, tv, v03)
        tv = v10; v10 = tl.where(do_swap, v13, v10); v13 = tl.where(do_swap, tv, v13)
        tv = v20; v20 = tl.where(do_swap, v23, v20); v23 = tl.where(do_swap, tv, v23)
        tv = v30; v30 = tl.where(do_swap, v33, v30); v33 = tl.where(do_swap, tv, v33)
        do_swap = s1 < s2
        s1, s2 = tl.where(do_swap, s2, s1), tl.where(do_swap, s1, s2)
        tv = v01; v01 = tl.where(do_swap, v02, v01); v02 = tl.where(do_swap, tv, v02)
        tv = v11; v11 = tl.where(do_swap, v12, v11); v12 = tl.where(do_swap, tv, v12)
        tv = v21; v21 = tl.where(do_swap, v22, v21); v22 = tl.where(do_swap, tv, v22)
        tv = v31; v31 = tl.where(do_swap, v32, v31); v32 = tl.where(do_swap, tv, v32)
        do_swap = s1 < s3
        s1, s3 = tl.where(do_swap, s3, s1), tl.where(do_swap, s1, s3)
        tv = v01; v01 = tl.where(do_swap, v03, v01); v03 = tl.where(do_swap, tv, v03)
        tv = v11; v11 = tl.where(do_swap, v13, v11); v13 = tl.where(do_swap, tv, v13)
        tv = v21; v21 = tl.where(do_swap, v23, v21); v23 = tl.where(do_swap, tv, v23)
        tv = v31; v31 = tl.where(do_swap, v33, v31); v33 = tl.where(do_swap, tv, v33)
        do_swap = s2 < s3
        s2, s3 = tl.where(do_swap, s3, s2), tl.where(do_swap, s2, s3)
        tv = v02; v02 = tl.where(do_swap, v03, v02); v03 = tl.where(do_swap, tv, v03)
        tv = v12; v12 = tl.where(do_swap, v13, v12); v13 = tl.where(do_swap, tv, v13)
        tv = v22; v22 = tl.where(do_swap, v23, v22); v23 = tl.where(do_swap, tv, v23)
        tv = v32; v32 = tl.where(do_swap, v33, v32); v33 = tl.where(do_swap, tv, v33)
        s_base = bid * 4
        tl.store(S_ptr + s_base + 0, s0); tl.store(S_ptr + s_base + 1, s1)
        tl.store(S_ptr + s_base + 2, s2); tl.store(S_ptr + s_base + 3, s3)
        # Vh = V^T — row r of Vh is column r of V.
        vh_base = bid * 16
        tl.store(Vh_ptr + vh_base + 0, v00); tl.store(Vh_ptr + vh_base + 1, v10)
        tl.store(Vh_ptr + vh_base + 2, v20); tl.store(Vh_ptr + vh_base + 3, v30)
        tl.store(Vh_ptr + vh_base + 4, v01); tl.store(Vh_ptr + vh_base + 5, v11)
        tl.store(Vh_ptr + vh_base + 6, v21); tl.store(Vh_ptr + vh_base + 7, v31)
        tl.store(Vh_ptr + vh_base + 8, v02); tl.store(Vh_ptr + vh_base + 9, v12)
        tl.store(Vh_ptr + vh_base + 10, v22); tl.store(Vh_ptr + vh_base + 11, v32)
        tl.store(Vh_ptr + vh_base + 12, v03); tl.store(Vh_ptr + vh_base + 13, v13)
        tl.store(Vh_ptr + vh_base + 14, v23); tl.store(Vh_ptr + vh_base + 15, v33)
        inv_s0 = 1.0 / (s0 + EPS); inv_s1 = 1.0 / (s1 + EPS)
        inv_s2 = 1.0 / (s2 + EPS); inv_s3 = 1.0 / (s3 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 4 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 4 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 4 + 2, mask=mask, other=0.0).to(DTYPE)
            a3 = tl.load(A_ptr + base + row_idx * 4 + 3, mask=mask, other=0.0).to(DTYPE)
            u0 = (a0*v00 + a1*v10 + a2*v20 + a3*v30) * inv_s0
            u1 = (a0*v01 + a1*v11 + a2*v21 + a3*v31) * inv_s1
            u2 = (a0*v02 + a1*v12 + a2*v22 + a3*v32) * inv_s2
            u3 = (a0*v03 + a1*v13 + a2*v23 + a3*v33) * inv_s3
            u_base = bid * M * 4
            tl.store(U_ptr + u_base + row_idx * 4 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 4 + 1, u1, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 4 + 2, u2, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 4 + 3, u3, mask=mask)

    # ── N=5: Cyclic Jacobi (10 pair sweep) ───────────────────────────────

    @triton.jit
    def _svd5_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        JACOBI_ITERS: tl.constexpr,
        DTYPE: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        base = bid * M * 5
        g00 = tl.zeros([], dtype=DTYPE); g01 = tl.zeros([], dtype=DTYPE); g02 = tl.zeros([], dtype=DTYPE); g03 = tl.zeros([], dtype=DTYPE); g04 = tl.zeros([], dtype=DTYPE)
        g11 = tl.zeros([], dtype=DTYPE); g12 = tl.zeros([], dtype=DTYPE); g13 = tl.zeros([], dtype=DTYPE); g14 = tl.zeros([], dtype=DTYPE)
        g22 = tl.zeros([], dtype=DTYPE); g23 = tl.zeros([], dtype=DTYPE); g24 = tl.zeros([], dtype=DTYPE)
        g33 = tl.zeros([], dtype=DTYPE); g34 = tl.zeros([], dtype=DTYPE)
        g44 = tl.zeros([], dtype=DTYPE)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 5 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 5 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 5 + 2, mask=mask, other=0.0).to(DTYPE)
            a3 = tl.load(A_ptr + base + row_idx * 5 + 3, mask=mask, other=0.0).to(DTYPE)
            a4 = tl.load(A_ptr + base + row_idx * 5 + 4, mask=mask, other=0.0).to(DTYPE)
            g00 += tl.sum(a0 * a0); g01 += tl.sum(a0 * a1); g02 += tl.sum(a0 * a2); g03 += tl.sum(a0 * a3); g04 += tl.sum(a0 * a4)
            g11 += tl.sum(a1 * a1); g12 += tl.sum(a1 * a2); g13 += tl.sum(a1 * a3); g14 += tl.sum(a1 * a4)
            g22 += tl.sum(a2 * a2); g23 += tl.sum(a2 * a3); g24 += tl.sum(a2 * a4)
            g33 += tl.sum(a3 * a3); g34 += tl.sum(a3 * a4)
            g44 += tl.sum(a4 * a4)
        v00 = tl.full([], 1.0, dtype=DTYPE); v01 = tl.zeros([], dtype=DTYPE); v02 = tl.zeros([], dtype=DTYPE); v03 = tl.zeros([], dtype=DTYPE); v04 = tl.zeros([], dtype=DTYPE)
        v10 = tl.zeros([], dtype=DTYPE); v11 = tl.full([], 1.0, dtype=DTYPE); v12 = tl.zeros([], dtype=DTYPE); v13 = tl.zeros([], dtype=DTYPE); v14 = tl.zeros([], dtype=DTYPE)
        v20 = tl.zeros([], dtype=DTYPE); v21 = tl.zeros([], dtype=DTYPE); v22 = tl.full([], 1.0, dtype=DTYPE); v23 = tl.zeros([], dtype=DTYPE); v24 = tl.zeros([], dtype=DTYPE)
        v30 = tl.zeros([], dtype=DTYPE); v31 = tl.zeros([], dtype=DTYPE); v32 = tl.zeros([], dtype=DTYPE); v33 = tl.full([], 1.0, dtype=DTYPE); v34 = tl.zeros([], dtype=DTYPE)
        v40 = tl.zeros([], dtype=DTYPE); v41 = tl.zeros([], dtype=DTYPE); v42 = tl.zeros([], dtype=DTYPE); v43 = tl.zeros([], dtype=DTYPE); v44 = tl.full([], 1.0, dtype=DTYPE)
        for _ in range(JACOBI_ITERS):
            # pair (0,1)
            off_diag = g01; diag_diff = g11 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g01 + s*s*g11; ng11 = s*s*g00 + 2.0*s*c*g01 + c*c*g11
            ng02 = c*g02 - s*g12; ng12 = s*g02 + c*g12
            ng03 = c*g03 - s*g13; ng13 = s*g03 + c*g13
            ng04 = c*g04 - s*g14; ng14 = s*g04 + c*g14
            g00 = ng00; g11 = ng11; g01 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g12 = ng12; g03 = ng03; g13 = ng13; g04 = ng04; g14 = ng14
            nv00 = c*v00 - s*v01; nv01 = s*v00 + c*v01
            nv10 = c*v10 - s*v11; nv11 = s*v10 + c*v11
            nv20 = c*v20 - s*v21; nv21 = s*v20 + c*v21
            nv30 = c*v30 - s*v31; nv31 = s*v30 + c*v31
            nv40 = c*v40 - s*v41; nv41 = s*v40 + c*v41
            v00 = nv00; v01 = nv01; v10 = nv10; v11 = nv11; v20 = nv20; v21 = nv21
            v30 = nv30; v31 = nv31; v40 = nv40; v41 = nv41
            # pair (0,2)
            off_diag = g02; diag_diff = g22 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g02 + s*s*g22; ng22 = s*s*g00 + 2.0*s*c*g02 + c*c*g22
            ng01 = c*g01 - s*g12; ng12 = s*g01 + c*g12
            ng03 = c*g03 - s*g23; ng23 = s*g03 + c*g23
            ng04 = c*g04 - s*g24; ng24 = s*g04 + c*g24
            g00 = ng00; g22 = ng22; g02 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g12 = ng12; g03 = ng03; g23 = ng23; g04 = ng04; g24 = ng24
            nv00 = c*v00 - s*v02; nv02 = s*v00 + c*v02
            nv10 = c*v10 - s*v12; nv12 = s*v10 + c*v12
            nv20 = c*v20 - s*v22; nv22 = s*v20 + c*v22
            nv30 = c*v30 - s*v32; nv32 = s*v30 + c*v32
            nv40 = c*v40 - s*v42; nv42 = s*v40 + c*v42
            v00 = nv00; v02 = nv02; v10 = nv10; v12 = nv12; v20 = nv20; v22 = nv22
            v30 = nv30; v32 = nv32; v40 = nv40; v42 = nv42
            # pair (0,3)
            off_diag = g03; diag_diff = g33 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g03 + s*s*g33; ng33 = s*s*g00 + 2.0*s*c*g03 + c*c*g33
            ng01 = c*g01 - s*g13; ng13 = s*g01 + c*g13
            ng02 = c*g02 - s*g23; ng23 = s*g02 + c*g23
            ng04 = c*g04 - s*g34; ng34 = s*g04 + c*g34
            g00 = ng00; g33 = ng33; g03 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g13 = ng13; g02 = ng02; g23 = ng23; g04 = ng04; g34 = ng34
            nv00 = c*v00 - s*v03; nv03 = s*v00 + c*v03
            nv10 = c*v10 - s*v13; nv13 = s*v10 + c*v13
            nv20 = c*v20 - s*v23; nv23 = s*v20 + c*v23
            nv30 = c*v30 - s*v33; nv33 = s*v30 + c*v33
            nv40 = c*v40 - s*v43; nv43 = s*v40 + c*v43
            v00 = nv00; v03 = nv03; v10 = nv10; v13 = nv13; v20 = nv20; v23 = nv23
            v30 = nv30; v33 = nv33; v40 = nv40; v43 = nv43
            # pair (0,4)
            off_diag = g04; diag_diff = g44 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g04 + s*s*g44; ng44 = s*s*g00 + 2.0*s*c*g04 + c*c*g44
            ng01 = c*g01 - s*g14; ng14 = s*g01 + c*g14
            ng02 = c*g02 - s*g24; ng24 = s*g02 + c*g24
            ng03 = c*g03 - s*g34; ng34 = s*g03 + c*g34
            g00 = ng00; g44 = ng44; g04 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g14 = ng14; g02 = ng02; g24 = ng24; g03 = ng03; g34 = ng34
            nv00 = c*v00 - s*v04; nv04 = s*v00 + c*v04
            nv10 = c*v10 - s*v14; nv14 = s*v10 + c*v14
            nv20 = c*v20 - s*v24; nv24 = s*v20 + c*v24
            nv30 = c*v30 - s*v34; nv34 = s*v30 + c*v34
            nv40 = c*v40 - s*v44; nv44 = s*v40 + c*v44
            v00 = nv00; v04 = nv04; v10 = nv10; v14 = nv14; v20 = nv20; v24 = nv24
            v30 = nv30; v34 = nv34; v40 = nv40; v44 = nv44
            # pair (1,2)
            off_diag = g12; diag_diff = g22 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g12 + s*s*g22; ng22 = s*s*g11 + 2.0*s*c*g12 + c*c*g22
            ng01 = c*g01 - s*g02; ng02 = s*g01 + c*g02
            ng13 = c*g13 - s*g23; ng23 = s*g13 + c*g23
            ng14 = c*g14 - s*g24; ng24 = s*g14 + c*g24
            g11 = ng11; g22 = ng22; g12 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g02 = ng02; g13 = ng13; g23 = ng23; g14 = ng14; g24 = ng24
            nv01 = c*v01 - s*v02; nv02 = s*v01 + c*v02
            nv11 = c*v11 - s*v12; nv12 = s*v11 + c*v12
            nv21 = c*v21 - s*v22; nv22 = s*v21 + c*v22
            nv31 = c*v31 - s*v32; nv32 = s*v31 + c*v32
            nv41 = c*v41 - s*v42; nv42 = s*v41 + c*v42
            v01 = nv01; v02 = nv02; v11 = nv11; v12 = nv12; v21 = nv21; v22 = nv22
            v31 = nv31; v32 = nv32; v41 = nv41; v42 = nv42
            # pair (1,3)
            off_diag = g13; diag_diff = g33 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g13 + s*s*g33; ng33 = s*s*g11 + 2.0*s*c*g13 + c*c*g33
            ng01 = c*g01 - s*g03; ng03 = s*g01 + c*g03
            ng12 = c*g12 - s*g23; ng23 = s*g12 + c*g23
            ng14 = c*g14 - s*g34; ng34 = s*g14 + c*g34
            g11 = ng11; g33 = ng33; g13 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g03 = ng03; g12 = ng12; g23 = ng23; g14 = ng14; g34 = ng34
            nv01 = c*v01 - s*v03; nv03 = s*v01 + c*v03
            nv11 = c*v11 - s*v13; nv13 = s*v11 + c*v13
            nv21 = c*v21 - s*v23; nv23 = s*v21 + c*v23
            nv31 = c*v31 - s*v33; nv33 = s*v31 + c*v33
            nv41 = c*v41 - s*v43; nv43 = s*v41 + c*v43
            v01 = nv01; v03 = nv03; v11 = nv11; v13 = nv13; v21 = nv21; v23 = nv23
            v31 = nv31; v33 = nv33; v41 = nv41; v43 = nv43
            # pair (1,4)
            off_diag = g14; diag_diff = g44 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g14 + s*s*g44; ng44 = s*s*g11 + 2.0*s*c*g14 + c*c*g44
            ng01 = c*g01 - s*g04; ng04 = s*g01 + c*g04
            ng12 = c*g12 - s*g24; ng24 = s*g12 + c*g24
            ng13 = c*g13 - s*g34; ng34 = s*g13 + c*g34
            g11 = ng11; g44 = ng44; g14 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g04 = ng04; g12 = ng12; g24 = ng24; g13 = ng13; g34 = ng34
            nv01 = c*v01 - s*v04; nv04 = s*v01 + c*v04
            nv11 = c*v11 - s*v14; nv14 = s*v11 + c*v14
            nv21 = c*v21 - s*v24; nv24 = s*v21 + c*v24
            nv31 = c*v31 - s*v34; nv34 = s*v31 + c*v34
            nv41 = c*v41 - s*v44; nv44 = s*v41 + c*v44
            v01 = nv01; v04 = nv04; v11 = nv11; v14 = nv14; v21 = nv21; v24 = nv24
            v31 = nv31; v34 = nv34; v41 = nv41; v44 = nv44
            # pair (2,3)
            off_diag = g23; diag_diff = g33 - g22; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng22 = c*c*g22 - 2.0*s*c*g23 + s*s*g33; ng33 = s*s*g22 + 2.0*s*c*g23 + c*c*g33
            ng02 = c*g02 - s*g03; ng03 = s*g02 + c*g03
            ng12 = c*g12 - s*g13; ng13 = s*g12 + c*g13
            ng24 = c*g24 - s*g34; ng34 = s*g24 + c*g34
            g22 = ng22; g33 = ng33; g23 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g03 = ng03; g12 = ng12; g13 = ng13; g24 = ng24; g34 = ng34
            nv02 = c*v02 - s*v03; nv03 = s*v02 + c*v03
            nv12 = c*v12 - s*v13; nv13 = s*v12 + c*v13
            nv22 = c*v22 - s*v23; nv23 = s*v22 + c*v23
            nv32 = c*v32 - s*v33; nv33 = s*v32 + c*v33
            nv42 = c*v42 - s*v43; nv43 = s*v42 + c*v43
            v02 = nv02; v03 = nv03; v12 = nv12; v13 = nv13; v22 = nv22; v23 = nv23
            v32 = nv32; v33 = nv33; v42 = nv42; v43 = nv43
            # pair (2,4)
            off_diag = g24; diag_diff = g44 - g22; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng22 = c*c*g22 - 2.0*s*c*g24 + s*s*g44; ng44 = s*s*g22 + 2.0*s*c*g24 + c*c*g44
            ng02 = c*g02 - s*g04; ng04 = s*g02 + c*g04
            ng12 = c*g12 - s*g14; ng14 = s*g12 + c*g14
            ng23 = c*g23 - s*g34; ng34 = s*g23 + c*g34
            g22 = ng22; g44 = ng44; g24 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g04 = ng04; g12 = ng12; g14 = ng14; g23 = ng23; g34 = ng34
            nv02 = c*v02 - s*v04; nv04 = s*v02 + c*v04
            nv12 = c*v12 - s*v14; nv14 = s*v12 + c*v14
            nv22 = c*v22 - s*v24; nv24 = s*v22 + c*v24
            nv32 = c*v32 - s*v34; nv34 = s*v32 + c*v34
            nv42 = c*v42 - s*v44; nv44 = s*v42 + c*v44
            v02 = nv02; v04 = nv04; v12 = nv12; v14 = nv14; v22 = nv22; v24 = nv24
            v32 = nv32; v34 = nv34; v42 = nv42; v44 = nv44
            # pair (3,4)
            off_diag = g34; diag_diff = g44 - g33; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng33 = c*c*g33 - 2.0*s*c*g34 + s*s*g44; ng44 = s*s*g33 + 2.0*s*c*g34 + c*c*g44
            ng03 = c*g03 - s*g04; ng04 = s*g03 + c*g04
            ng13 = c*g13 - s*g14; ng14 = s*g13 + c*g14
            ng23 = c*g23 - s*g24; ng24 = s*g23 + c*g24
            g33 = ng33; g44 = ng44; g34 = tl.zeros([], dtype=DTYPE)
            g03 = ng03; g04 = ng04; g13 = ng13; g14 = ng14; g23 = ng23; g24 = ng24
            nv03 = c*v03 - s*v04; nv04 = s*v03 + c*v04
            nv13 = c*v13 - s*v14; nv14 = s*v13 + c*v14
            nv23 = c*v23 - s*v24; nv24 = s*v23 + c*v24
            nv33 = c*v33 - s*v34; nv34 = s*v33 + c*v34
            nv43 = c*v43 - s*v44; nv44 = s*v43 + c*v44
            v03 = nv03; v04 = nv04; v13 = nv13; v14 = nv14; v23 = nv23; v24 = nv24
            v33 = nv33; v34 = nv34; v43 = nv43; v44 = nv44
        s0 = tl.sqrt(tl.maximum(g00, EPS))
        s1 = tl.sqrt(tl.maximum(g11, EPS))
        s2 = tl.sqrt(tl.maximum(g22, EPS))
        s3 = tl.sqrt(tl.maximum(g33, EPS))
        s4 = tl.sqrt(tl.maximum(g44, EPS))
        # Selection sort descending: swap V columns too.
        # swap(0,1)
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00; v00 = tl.where(do_swap, v01, v00); v01 = tl.where(do_swap, tv, v01)
        tv = v10; v10 = tl.where(do_swap, v11, v10); v11 = tl.where(do_swap, tv, v11)
        tv = v20; v20 = tl.where(do_swap, v21, v20); v21 = tl.where(do_swap, tv, v21)
        tv = v30; v30 = tl.where(do_swap, v31, v30); v31 = tl.where(do_swap, tv, v31)
        tv = v40; v40 = tl.where(do_swap, v41, v40); v41 = tl.where(do_swap, tv, v41)
        # swap(0,2)
        do_swap = s0 < s2
        s0, s2 = tl.where(do_swap, s2, s0), tl.where(do_swap, s0, s2)
        tv = v00; v00 = tl.where(do_swap, v02, v00); v02 = tl.where(do_swap, tv, v02)
        tv = v10; v10 = tl.where(do_swap, v12, v10); v12 = tl.where(do_swap, tv, v12)
        tv = v20; v20 = tl.where(do_swap, v22, v20); v22 = tl.where(do_swap, tv, v22)
        tv = v30; v30 = tl.where(do_swap, v32, v30); v32 = tl.where(do_swap, tv, v32)
        tv = v40; v40 = tl.where(do_swap, v42, v40); v42 = tl.where(do_swap, tv, v42)
        # swap(0,3)
        do_swap = s0 < s3
        s0, s3 = tl.where(do_swap, s3, s0), tl.where(do_swap, s0, s3)
        tv = v00; v00 = tl.where(do_swap, v03, v00); v03 = tl.where(do_swap, tv, v03)
        tv = v10; v10 = tl.where(do_swap, v13, v10); v13 = tl.where(do_swap, tv, v13)
        tv = v20; v20 = tl.where(do_swap, v23, v20); v23 = tl.where(do_swap, tv, v23)
        tv = v30; v30 = tl.where(do_swap, v33, v30); v33 = tl.where(do_swap, tv, v33)
        tv = v40; v40 = tl.where(do_swap, v43, v40); v43 = tl.where(do_swap, tv, v43)
        # swap(0,4)
        do_swap = s0 < s4
        s0, s4 = tl.where(do_swap, s4, s0), tl.where(do_swap, s0, s4)
        tv = v00; v00 = tl.where(do_swap, v04, v00); v04 = tl.where(do_swap, tv, v04)
        tv = v10; v10 = tl.where(do_swap, v14, v10); v14 = tl.where(do_swap, tv, v14)
        tv = v20; v20 = tl.where(do_swap, v24, v20); v24 = tl.where(do_swap, tv, v24)
        tv = v30; v30 = tl.where(do_swap, v34, v30); v34 = tl.where(do_swap, tv, v34)
        tv = v40; v40 = tl.where(do_swap, v44, v40); v44 = tl.where(do_swap, tv, v44)
        # swap(1,2)
        do_swap = s1 < s2
        s1, s2 = tl.where(do_swap, s2, s1), tl.where(do_swap, s1, s2)
        tv = v01; v01 = tl.where(do_swap, v02, v01); v02 = tl.where(do_swap, tv, v02)
        tv = v11; v11 = tl.where(do_swap, v12, v11); v12 = tl.where(do_swap, tv, v12)
        tv = v21; v21 = tl.where(do_swap, v22, v21); v22 = tl.where(do_swap, tv, v22)
        tv = v31; v31 = tl.where(do_swap, v32, v31); v32 = tl.where(do_swap, tv, v32)
        tv = v41; v41 = tl.where(do_swap, v42, v41); v42 = tl.where(do_swap, tv, v42)
        # swap(1,3)
        do_swap = s1 < s3
        s1, s3 = tl.where(do_swap, s3, s1), tl.where(do_swap, s1, s3)
        tv = v01; v01 = tl.where(do_swap, v03, v01); v03 = tl.where(do_swap, tv, v03)
        tv = v11; v11 = tl.where(do_swap, v13, v11); v13 = tl.where(do_swap, tv, v13)
        tv = v21; v21 = tl.where(do_swap, v23, v21); v23 = tl.where(do_swap, tv, v23)
        tv = v31; v31 = tl.where(do_swap, v33, v31); v33 = tl.where(do_swap, tv, v33)
        tv = v41; v41 = tl.where(do_swap, v43, v41); v43 = tl.where(do_swap, tv, v43)
        # swap(1,4)
        do_swap = s1 < s4
        s1, s4 = tl.where(do_swap, s4, s1), tl.where(do_swap, s1, s4)
        tv = v01; v01 = tl.where(do_swap, v04, v01); v04 = tl.where(do_swap, tv, v04)
        tv = v11; v11 = tl.where(do_swap, v14, v11); v14 = tl.where(do_swap, tv, v14)
        tv = v21; v21 = tl.where(do_swap, v24, v21); v24 = tl.where(do_swap, tv, v24)
        tv = v31; v31 = tl.where(do_swap, v34, v31); v34 = tl.where(do_swap, tv, v34)
        tv = v41; v41 = tl.where(do_swap, v44, v41); v44 = tl.where(do_swap, tv, v44)
        # swap(2,3)
        do_swap = s2 < s3
        s2, s3 = tl.where(do_swap, s3, s2), tl.where(do_swap, s2, s3)
        tv = v02; v02 = tl.where(do_swap, v03, v02); v03 = tl.where(do_swap, tv, v03)
        tv = v12; v12 = tl.where(do_swap, v13, v12); v13 = tl.where(do_swap, tv, v13)
        tv = v22; v22 = tl.where(do_swap, v23, v22); v23 = tl.where(do_swap, tv, v23)
        tv = v32; v32 = tl.where(do_swap, v33, v32); v33 = tl.where(do_swap, tv, v33)
        tv = v42; v42 = tl.where(do_swap, v43, v42); v43 = tl.where(do_swap, tv, v43)
        # swap(2,4)
        do_swap = s2 < s4
        s2, s4 = tl.where(do_swap, s4, s2), tl.where(do_swap, s2, s4)
        tv = v02; v02 = tl.where(do_swap, v04, v02); v04 = tl.where(do_swap, tv, v04)
        tv = v12; v12 = tl.where(do_swap, v14, v12); v14 = tl.where(do_swap, tv, v14)
        tv = v22; v22 = tl.where(do_swap, v24, v22); v24 = tl.where(do_swap, tv, v24)
        tv = v32; v32 = tl.where(do_swap, v34, v32); v34 = tl.where(do_swap, tv, v34)
        tv = v42; v42 = tl.where(do_swap, v44, v42); v44 = tl.where(do_swap, tv, v44)
        # swap(3,4)
        do_swap = s3 < s4
        s3, s4 = tl.where(do_swap, s4, s3), tl.where(do_swap, s3, s4)
        tv = v03; v03 = tl.where(do_swap, v04, v03); v04 = tl.where(do_swap, tv, v04)
        tv = v13; v13 = tl.where(do_swap, v14, v13); v14 = tl.where(do_swap, tv, v14)
        tv = v23; v23 = tl.where(do_swap, v24, v23); v24 = tl.where(do_swap, tv, v24)
        tv = v33; v33 = tl.where(do_swap, v34, v33); v34 = tl.where(do_swap, tv, v34)
        tv = v43; v43 = tl.where(do_swap, v44, v43); v44 = tl.where(do_swap, tv, v44)
        s_base = bid * 5
        tl.store(S_ptr + s_base + 0, s0); tl.store(S_ptr + s_base + 1, s1)
        tl.store(S_ptr + s_base + 2, s2); tl.store(S_ptr + s_base + 3, s3)
        tl.store(S_ptr + s_base + 4, s4)
        vh_base = bid * 25
        tl.store(Vh_ptr + vh_base + 0, v00); tl.store(Vh_ptr + vh_base + 1, v10)
        tl.store(Vh_ptr + vh_base + 2, v20); tl.store(Vh_ptr + vh_base + 3, v30); tl.store(Vh_ptr + vh_base + 4, v40)
        tl.store(Vh_ptr + vh_base + 5, v01); tl.store(Vh_ptr + vh_base + 6, v11)
        tl.store(Vh_ptr + vh_base + 7, v21); tl.store(Vh_ptr + vh_base + 8, v31); tl.store(Vh_ptr + vh_base + 9, v41)
        tl.store(Vh_ptr + vh_base + 10, v02); tl.store(Vh_ptr + vh_base + 11, v12)
        tl.store(Vh_ptr + vh_base + 12, v22); tl.store(Vh_ptr + vh_base + 13, v32); tl.store(Vh_ptr + vh_base + 14, v42)
        tl.store(Vh_ptr + vh_base + 15, v03); tl.store(Vh_ptr + vh_base + 16, v13)
        tl.store(Vh_ptr + vh_base + 17, v23); tl.store(Vh_ptr + vh_base + 18, v33); tl.store(Vh_ptr + vh_base + 19, v43)
        tl.store(Vh_ptr + vh_base + 20, v04); tl.store(Vh_ptr + vh_base + 21, v14)
        tl.store(Vh_ptr + vh_base + 22, v24); tl.store(Vh_ptr + vh_base + 23, v34); tl.store(Vh_ptr + vh_base + 24, v44)
        inv_s0 = 1.0 / (s0 + EPS); inv_s1 = 1.0 / (s1 + EPS)
        inv_s2 = 1.0 / (s2 + EPS); inv_s3 = 1.0 / (s3 + EPS); inv_s4 = 1.0 / (s4 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 5 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 5 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 5 + 2, mask=mask, other=0.0).to(DTYPE)
            a3 = tl.load(A_ptr + base + row_idx * 5 + 3, mask=mask, other=0.0).to(DTYPE)
            a4 = tl.load(A_ptr + base + row_idx * 5 + 4, mask=mask, other=0.0).to(DTYPE)
            u0 = (a0*v00 + a1*v10 + a2*v20 + a3*v30 + a4*v40) * inv_s0
            u1 = (a0*v01 + a1*v11 + a2*v21 + a3*v31 + a4*v41) * inv_s1
            u2 = (a0*v02 + a1*v12 + a2*v22 + a3*v32 + a4*v42) * inv_s2
            u3 = (a0*v03 + a1*v13 + a2*v23 + a3*v33 + a4*v43) * inv_s3
            u4 = (a0*v04 + a1*v14 + a2*v24 + a3*v34 + a4*v44) * inv_s4
            u_base = bid * M * 5
            tl.store(U_ptr + u_base + row_idx * 5 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 5 + 1, u1, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 5 + 2, u2, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 5 + 3, u3, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 5 + 4, u4, mask=mask)

    # ── N=6: Cyclic Jacobi (15 pair sweep) ──

    @triton.jit
    def _svd6_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        JACOBI_ITERS: tl.constexpr,
        DTYPE: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        base = bid * M * 6
        # Gram accumulators (upper triangle).
        g00 = tl.zeros([], dtype=DTYPE); g01 = tl.zeros([], dtype=DTYPE); g02 = tl.zeros([], dtype=DTYPE); g03 = tl.zeros([], dtype=DTYPE); g04 = tl.zeros([], dtype=DTYPE); g05 = tl.zeros([], dtype=DTYPE)
        g11 = tl.zeros([], dtype=DTYPE); g12 = tl.zeros([], dtype=DTYPE); g13 = tl.zeros([], dtype=DTYPE); g14 = tl.zeros([], dtype=DTYPE); g15 = tl.zeros([], dtype=DTYPE)
        g22 = tl.zeros([], dtype=DTYPE); g23 = tl.zeros([], dtype=DTYPE); g24 = tl.zeros([], dtype=DTYPE); g25 = tl.zeros([], dtype=DTYPE)
        g33 = tl.zeros([], dtype=DTYPE); g34 = tl.zeros([], dtype=DTYPE); g35 = tl.zeros([], dtype=DTYPE)
        g44 = tl.zeros([], dtype=DTYPE); g45 = tl.zeros([], dtype=DTYPE)
        g55 = tl.zeros([], dtype=DTYPE)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 6 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 6 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 6 + 2, mask=mask, other=0.0).to(DTYPE)
            a3 = tl.load(A_ptr + base + row_idx * 6 + 3, mask=mask, other=0.0).to(DTYPE)
            a4 = tl.load(A_ptr + base + row_idx * 6 + 4, mask=mask, other=0.0).to(DTYPE)
            a5 = tl.load(A_ptr + base + row_idx * 6 + 5, mask=mask, other=0.0).to(DTYPE)
            g00 += tl.sum(a0 * a0); g01 += tl.sum(a0 * a1); g02 += tl.sum(a0 * a2); g03 += tl.sum(a0 * a3); g04 += tl.sum(a0 * a4); g05 += tl.sum(a0 * a5)
            g11 += tl.sum(a1 * a1); g12 += tl.sum(a1 * a2); g13 += tl.sum(a1 * a3); g14 += tl.sum(a1 * a4); g15 += tl.sum(a1 * a5)
            g22 += tl.sum(a2 * a2); g23 += tl.sum(a2 * a3); g24 += tl.sum(a2 * a4); g25 += tl.sum(a2 * a5)
            g33 += tl.sum(a3 * a3); g34 += tl.sum(a3 * a4); g35 += tl.sum(a3 * a5)
            g44 += tl.sum(a4 * a4); g45 += tl.sum(a4 * a5)
            g55 += tl.sum(a5 * a5)
        # V = I_N (column-major: v[row][col]).
        v00 = tl.full([], 1.0, dtype=DTYPE); v01 = tl.zeros([], dtype=DTYPE); v02 = tl.zeros([], dtype=DTYPE); v03 = tl.zeros([], dtype=DTYPE); v04 = tl.zeros([], dtype=DTYPE); v05 = tl.zeros([], dtype=DTYPE)
        v10 = tl.zeros([], dtype=DTYPE); v11 = tl.full([], 1.0, dtype=DTYPE); v12 = tl.zeros([], dtype=DTYPE); v13 = tl.zeros([], dtype=DTYPE); v14 = tl.zeros([], dtype=DTYPE); v15 = tl.zeros([], dtype=DTYPE)
        v20 = tl.zeros([], dtype=DTYPE); v21 = tl.zeros([], dtype=DTYPE); v22 = tl.full([], 1.0, dtype=DTYPE); v23 = tl.zeros([], dtype=DTYPE); v24 = tl.zeros([], dtype=DTYPE); v25 = tl.zeros([], dtype=DTYPE)
        v30 = tl.zeros([], dtype=DTYPE); v31 = tl.zeros([], dtype=DTYPE); v32 = tl.zeros([], dtype=DTYPE); v33 = tl.full([], 1.0, dtype=DTYPE); v34 = tl.zeros([], dtype=DTYPE); v35 = tl.zeros([], dtype=DTYPE)
        v40 = tl.zeros([], dtype=DTYPE); v41 = tl.zeros([], dtype=DTYPE); v42 = tl.zeros([], dtype=DTYPE); v43 = tl.zeros([], dtype=DTYPE); v44 = tl.full([], 1.0, dtype=DTYPE); v45 = tl.zeros([], dtype=DTYPE)
        v50 = tl.zeros([], dtype=DTYPE); v51 = tl.zeros([], dtype=DTYPE); v52 = tl.zeros([], dtype=DTYPE); v53 = tl.zeros([], dtype=DTYPE); v54 = tl.zeros([], dtype=DTYPE); v55 = tl.full([], 1.0, dtype=DTYPE)
        for _ in range(JACOBI_ITERS):
            # pair (0,1)
            off_diag = g01; diag_diff = g11 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g01 + s*s*g11; ng11 = s*s*g00 + 2.0*s*c*g01 + c*c*g11
            ng02 = c*g02 - s*g12; ng12 = s*g02 + c*g12
            ng03 = c*g03 - s*g13; ng13 = s*g03 + c*g13
            ng04 = c*g04 - s*g14; ng14 = s*g04 + c*g14
            ng05 = c*g05 - s*g15; ng15 = s*g05 + c*g15
            g00 = ng00; g11 = ng11; g01 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g12 = ng12; g03 = ng03; g13 = ng13; g04 = ng04; g14 = ng14; g05 = ng05; g15 = ng15
            nv00 = c*v00 - s*v01; nv01 = s*v00 + c*v01
            nv10 = c*v10 - s*v11; nv11 = s*v10 + c*v11
            nv20 = c*v20 - s*v21; nv21 = s*v20 + c*v21
            nv30 = c*v30 - s*v31; nv31 = s*v30 + c*v31
            nv40 = c*v40 - s*v41; nv41 = s*v40 + c*v41
            nv50 = c*v50 - s*v51; nv51 = s*v50 + c*v51
            v00 = nv00; v01 = nv01; v10 = nv10; v11 = nv11; v20 = nv20; v21 = nv21; v30 = nv30; v31 = nv31; v40 = nv40; v41 = nv41; v50 = nv50; v51 = nv51
            # pair (0,2)
            off_diag = g02; diag_diff = g22 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g02 + s*s*g22; ng22 = s*s*g00 + 2.0*s*c*g02 + c*c*g22
            ng01 = c*g01 - s*g12; ng12 = s*g01 + c*g12
            ng03 = c*g03 - s*g23; ng23 = s*g03 + c*g23
            ng04 = c*g04 - s*g24; ng24 = s*g04 + c*g24
            ng05 = c*g05 - s*g25; ng25 = s*g05 + c*g25
            g00 = ng00; g22 = ng22; g02 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g12 = ng12; g03 = ng03; g23 = ng23; g04 = ng04; g24 = ng24; g05 = ng05; g25 = ng25
            nv00 = c*v00 - s*v02; nv02 = s*v00 + c*v02
            nv10 = c*v10 - s*v12; nv12 = s*v10 + c*v12
            nv20 = c*v20 - s*v22; nv22 = s*v20 + c*v22
            nv30 = c*v30 - s*v32; nv32 = s*v30 + c*v32
            nv40 = c*v40 - s*v42; nv42 = s*v40 + c*v42
            nv50 = c*v50 - s*v52; nv52 = s*v50 + c*v52
            v00 = nv00; v02 = nv02; v10 = nv10; v12 = nv12; v20 = nv20; v22 = nv22; v30 = nv30; v32 = nv32; v40 = nv40; v42 = nv42; v50 = nv50; v52 = nv52
            # pair (0,3)
            off_diag = g03; diag_diff = g33 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g03 + s*s*g33; ng33 = s*s*g00 + 2.0*s*c*g03 + c*c*g33
            ng01 = c*g01 - s*g13; ng13 = s*g01 + c*g13
            ng02 = c*g02 - s*g23; ng23 = s*g02 + c*g23
            ng04 = c*g04 - s*g34; ng34 = s*g04 + c*g34
            ng05 = c*g05 - s*g35; ng35 = s*g05 + c*g35
            g00 = ng00; g33 = ng33; g03 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g13 = ng13; g02 = ng02; g23 = ng23; g04 = ng04; g34 = ng34; g05 = ng05; g35 = ng35
            nv00 = c*v00 - s*v03; nv03 = s*v00 + c*v03
            nv10 = c*v10 - s*v13; nv13 = s*v10 + c*v13
            nv20 = c*v20 - s*v23; nv23 = s*v20 + c*v23
            nv30 = c*v30 - s*v33; nv33 = s*v30 + c*v33
            nv40 = c*v40 - s*v43; nv43 = s*v40 + c*v43
            nv50 = c*v50 - s*v53; nv53 = s*v50 + c*v53
            v00 = nv00; v03 = nv03; v10 = nv10; v13 = nv13; v20 = nv20; v23 = nv23; v30 = nv30; v33 = nv33; v40 = nv40; v43 = nv43; v50 = nv50; v53 = nv53
            # pair (0,4)
            off_diag = g04; diag_diff = g44 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g04 + s*s*g44; ng44 = s*s*g00 + 2.0*s*c*g04 + c*c*g44
            ng01 = c*g01 - s*g14; ng14 = s*g01 + c*g14
            ng02 = c*g02 - s*g24; ng24 = s*g02 + c*g24
            ng03 = c*g03 - s*g34; ng34 = s*g03 + c*g34
            ng05 = c*g05 - s*g45; ng45 = s*g05 + c*g45
            g00 = ng00; g44 = ng44; g04 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g14 = ng14; g02 = ng02; g24 = ng24; g03 = ng03; g34 = ng34; g05 = ng05; g45 = ng45
            nv00 = c*v00 - s*v04; nv04 = s*v00 + c*v04
            nv10 = c*v10 - s*v14; nv14 = s*v10 + c*v14
            nv20 = c*v20 - s*v24; nv24 = s*v20 + c*v24
            nv30 = c*v30 - s*v34; nv34 = s*v30 + c*v34
            nv40 = c*v40 - s*v44; nv44 = s*v40 + c*v44
            nv50 = c*v50 - s*v54; nv54 = s*v50 + c*v54
            v00 = nv00; v04 = nv04; v10 = nv10; v14 = nv14; v20 = nv20; v24 = nv24; v30 = nv30; v34 = nv34; v40 = nv40; v44 = nv44; v50 = nv50; v54 = nv54
            # pair (0,5)
            off_diag = g05; diag_diff = g55 - g00; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g05 + s*s*g55; ng55 = s*s*g00 + 2.0*s*c*g05 + c*c*g55
            ng01 = c*g01 - s*g15; ng15 = s*g01 + c*g15
            ng02 = c*g02 - s*g25; ng25 = s*g02 + c*g25
            ng03 = c*g03 - s*g35; ng35 = s*g03 + c*g35
            ng04 = c*g04 - s*g45; ng45 = s*g04 + c*g45
            g00 = ng00; g55 = ng55; g05 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g15 = ng15; g02 = ng02; g25 = ng25; g03 = ng03; g35 = ng35; g04 = ng04; g45 = ng45
            nv00 = c*v00 - s*v05; nv05 = s*v00 + c*v05
            nv10 = c*v10 - s*v15; nv15 = s*v10 + c*v15
            nv20 = c*v20 - s*v25; nv25 = s*v20 + c*v25
            nv30 = c*v30 - s*v35; nv35 = s*v30 + c*v35
            nv40 = c*v40 - s*v45; nv45 = s*v40 + c*v45
            nv50 = c*v50 - s*v55; nv55 = s*v50 + c*v55
            v00 = nv00; v05 = nv05; v10 = nv10; v15 = nv15; v20 = nv20; v25 = nv25; v30 = nv30; v35 = nv35; v40 = nv40; v45 = nv45; v50 = nv50; v55 = nv55
            # pair (1,2)
            off_diag = g12; diag_diff = g22 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g12 + s*s*g22; ng22 = s*s*g11 + 2.0*s*c*g12 + c*c*g22
            ng01 = c*g01 - s*g02; ng02 = s*g01 + c*g02
            ng13 = c*g13 - s*g23; ng23 = s*g13 + c*g23
            ng14 = c*g14 - s*g24; ng24 = s*g14 + c*g24
            ng15 = c*g15 - s*g25; ng25 = s*g15 + c*g25
            g11 = ng11; g22 = ng22; g12 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g02 = ng02; g13 = ng13; g23 = ng23; g14 = ng14; g24 = ng24; g15 = ng15; g25 = ng25
            nv01 = c*v01 - s*v02; nv02 = s*v01 + c*v02
            nv11 = c*v11 - s*v12; nv12 = s*v11 + c*v12
            nv21 = c*v21 - s*v22; nv22 = s*v21 + c*v22
            nv31 = c*v31 - s*v32; nv32 = s*v31 + c*v32
            nv41 = c*v41 - s*v42; nv42 = s*v41 + c*v42
            nv51 = c*v51 - s*v52; nv52 = s*v51 + c*v52
            v01 = nv01; v02 = nv02; v11 = nv11; v12 = nv12; v21 = nv21; v22 = nv22; v31 = nv31; v32 = nv32; v41 = nv41; v42 = nv42; v51 = nv51; v52 = nv52
            # pair (1,3)
            off_diag = g13; diag_diff = g33 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g13 + s*s*g33; ng33 = s*s*g11 + 2.0*s*c*g13 + c*c*g33
            ng01 = c*g01 - s*g03; ng03 = s*g01 + c*g03
            ng12 = c*g12 - s*g23; ng23 = s*g12 + c*g23
            ng14 = c*g14 - s*g34; ng34 = s*g14 + c*g34
            ng15 = c*g15 - s*g35; ng35 = s*g15 + c*g35
            g11 = ng11; g33 = ng33; g13 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g03 = ng03; g12 = ng12; g23 = ng23; g14 = ng14; g34 = ng34; g15 = ng15; g35 = ng35
            nv01 = c*v01 - s*v03; nv03 = s*v01 + c*v03
            nv11 = c*v11 - s*v13; nv13 = s*v11 + c*v13
            nv21 = c*v21 - s*v23; nv23 = s*v21 + c*v23
            nv31 = c*v31 - s*v33; nv33 = s*v31 + c*v33
            nv41 = c*v41 - s*v43; nv43 = s*v41 + c*v43
            nv51 = c*v51 - s*v53; nv53 = s*v51 + c*v53
            v01 = nv01; v03 = nv03; v11 = nv11; v13 = nv13; v21 = nv21; v23 = nv23; v31 = nv31; v33 = nv33; v41 = nv41; v43 = nv43; v51 = nv51; v53 = nv53
            # pair (1,4)
            off_diag = g14; diag_diff = g44 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g14 + s*s*g44; ng44 = s*s*g11 + 2.0*s*c*g14 + c*c*g44
            ng01 = c*g01 - s*g04; ng04 = s*g01 + c*g04
            ng12 = c*g12 - s*g24; ng24 = s*g12 + c*g24
            ng13 = c*g13 - s*g34; ng34 = s*g13 + c*g34
            ng15 = c*g15 - s*g45; ng45 = s*g15 + c*g45
            g11 = ng11; g44 = ng44; g14 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g04 = ng04; g12 = ng12; g24 = ng24; g13 = ng13; g34 = ng34; g15 = ng15; g45 = ng45
            nv01 = c*v01 - s*v04; nv04 = s*v01 + c*v04
            nv11 = c*v11 - s*v14; nv14 = s*v11 + c*v14
            nv21 = c*v21 - s*v24; nv24 = s*v21 + c*v24
            nv31 = c*v31 - s*v34; nv34 = s*v31 + c*v34
            nv41 = c*v41 - s*v44; nv44 = s*v41 + c*v44
            nv51 = c*v51 - s*v54; nv54 = s*v51 + c*v54
            v01 = nv01; v04 = nv04; v11 = nv11; v14 = nv14; v21 = nv21; v24 = nv24; v31 = nv31; v34 = nv34; v41 = nv41; v44 = nv44; v51 = nv51; v54 = nv54
            # pair (1,5)
            off_diag = g15; diag_diff = g55 - g11; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g15 + s*s*g55; ng55 = s*s*g11 + 2.0*s*c*g15 + c*c*g55
            ng01 = c*g01 - s*g05; ng05 = s*g01 + c*g05
            ng12 = c*g12 - s*g25; ng25 = s*g12 + c*g25
            ng13 = c*g13 - s*g35; ng35 = s*g13 + c*g35
            ng14 = c*g14 - s*g45; ng45 = s*g14 + c*g45
            g11 = ng11; g55 = ng55; g15 = tl.zeros([], dtype=DTYPE)
            g01 = ng01; g05 = ng05; g12 = ng12; g25 = ng25; g13 = ng13; g35 = ng35; g14 = ng14; g45 = ng45
            nv01 = c*v01 - s*v05; nv05 = s*v01 + c*v05
            nv11 = c*v11 - s*v15; nv15 = s*v11 + c*v15
            nv21 = c*v21 - s*v25; nv25 = s*v21 + c*v25
            nv31 = c*v31 - s*v35; nv35 = s*v31 + c*v35
            nv41 = c*v41 - s*v45; nv45 = s*v41 + c*v45
            nv51 = c*v51 - s*v55; nv55 = s*v51 + c*v55
            v01 = nv01; v05 = nv05; v11 = nv11; v15 = nv15; v21 = nv21; v25 = nv25; v31 = nv31; v35 = nv35; v41 = nv41; v45 = nv45; v51 = nv51; v55 = nv55
            # pair (2,3)
            off_diag = g23; diag_diff = g33 - g22; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng22 = c*c*g22 - 2.0*s*c*g23 + s*s*g33; ng33 = s*s*g22 + 2.0*s*c*g23 + c*c*g33
            ng02 = c*g02 - s*g03; ng03 = s*g02 + c*g03
            ng12 = c*g12 - s*g13; ng13 = s*g12 + c*g13
            ng24 = c*g24 - s*g34; ng34 = s*g24 + c*g34
            ng25 = c*g25 - s*g35; ng35 = s*g25 + c*g35
            g22 = ng22; g33 = ng33; g23 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g03 = ng03; g12 = ng12; g13 = ng13; g24 = ng24; g34 = ng34; g25 = ng25; g35 = ng35
            nv02 = c*v02 - s*v03; nv03 = s*v02 + c*v03
            nv12 = c*v12 - s*v13; nv13 = s*v12 + c*v13
            nv22 = c*v22 - s*v23; nv23 = s*v22 + c*v23
            nv32 = c*v32 - s*v33; nv33 = s*v32 + c*v33
            nv42 = c*v42 - s*v43; nv43 = s*v42 + c*v43
            nv52 = c*v52 - s*v53; nv53 = s*v52 + c*v53
            v02 = nv02; v03 = nv03; v12 = nv12; v13 = nv13; v22 = nv22; v23 = nv23; v32 = nv32; v33 = nv33; v42 = nv42; v43 = nv43; v52 = nv52; v53 = nv53
            # pair (2,4)
            off_diag = g24; diag_diff = g44 - g22; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng22 = c*c*g22 - 2.0*s*c*g24 + s*s*g44; ng44 = s*s*g22 + 2.0*s*c*g24 + c*c*g44
            ng02 = c*g02 - s*g04; ng04 = s*g02 + c*g04
            ng12 = c*g12 - s*g14; ng14 = s*g12 + c*g14
            ng23 = c*g23 - s*g34; ng34 = s*g23 + c*g34
            ng25 = c*g25 - s*g45; ng45 = s*g25 + c*g45
            g22 = ng22; g44 = ng44; g24 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g04 = ng04; g12 = ng12; g14 = ng14; g23 = ng23; g34 = ng34; g25 = ng25; g45 = ng45
            nv02 = c*v02 - s*v04; nv04 = s*v02 + c*v04
            nv12 = c*v12 - s*v14; nv14 = s*v12 + c*v14
            nv22 = c*v22 - s*v24; nv24 = s*v22 + c*v24
            nv32 = c*v32 - s*v34; nv34 = s*v32 + c*v34
            nv42 = c*v42 - s*v44; nv44 = s*v42 + c*v44
            nv52 = c*v52 - s*v54; nv54 = s*v52 + c*v54
            v02 = nv02; v04 = nv04; v12 = nv12; v14 = nv14; v22 = nv22; v24 = nv24; v32 = nv32; v34 = nv34; v42 = nv42; v44 = nv44; v52 = nv52; v54 = nv54
            # pair (2,5)
            off_diag = g25; diag_diff = g55 - g22; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng22 = c*c*g22 - 2.0*s*c*g25 + s*s*g55; ng55 = s*s*g22 + 2.0*s*c*g25 + c*c*g55
            ng02 = c*g02 - s*g05; ng05 = s*g02 + c*g05
            ng12 = c*g12 - s*g15; ng15 = s*g12 + c*g15
            ng23 = c*g23 - s*g35; ng35 = s*g23 + c*g35
            ng24 = c*g24 - s*g45; ng45 = s*g24 + c*g45
            g22 = ng22; g55 = ng55; g25 = tl.zeros([], dtype=DTYPE)
            g02 = ng02; g05 = ng05; g12 = ng12; g15 = ng15; g23 = ng23; g35 = ng35; g24 = ng24; g45 = ng45
            nv02 = c*v02 - s*v05; nv05 = s*v02 + c*v05
            nv12 = c*v12 - s*v15; nv15 = s*v12 + c*v15
            nv22 = c*v22 - s*v25; nv25 = s*v22 + c*v25
            nv32 = c*v32 - s*v35; nv35 = s*v32 + c*v35
            nv42 = c*v42 - s*v45; nv45 = s*v42 + c*v45
            nv52 = c*v52 - s*v55; nv55 = s*v52 + c*v55
            v02 = nv02; v05 = nv05; v12 = nv12; v15 = nv15; v22 = nv22; v25 = nv25; v32 = nv32; v35 = nv35; v42 = nv42; v45 = nv45; v52 = nv52; v55 = nv55
            # pair (3,4)
            off_diag = g34; diag_diff = g44 - g33; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng33 = c*c*g33 - 2.0*s*c*g34 + s*s*g44; ng44 = s*s*g33 + 2.0*s*c*g34 + c*c*g44
            ng03 = c*g03 - s*g04; ng04 = s*g03 + c*g04
            ng13 = c*g13 - s*g14; ng14 = s*g13 + c*g14
            ng23 = c*g23 - s*g24; ng24 = s*g23 + c*g24
            ng35 = c*g35 - s*g45; ng45 = s*g35 + c*g45
            g33 = ng33; g44 = ng44; g34 = tl.zeros([], dtype=DTYPE)
            g03 = ng03; g04 = ng04; g13 = ng13; g14 = ng14; g23 = ng23; g24 = ng24; g35 = ng35; g45 = ng45
            nv03 = c*v03 - s*v04; nv04 = s*v03 + c*v04
            nv13 = c*v13 - s*v14; nv14 = s*v13 + c*v14
            nv23 = c*v23 - s*v24; nv24 = s*v23 + c*v24
            nv33 = c*v33 - s*v34; nv34 = s*v33 + c*v34
            nv43 = c*v43 - s*v44; nv44 = s*v43 + c*v44
            nv53 = c*v53 - s*v54; nv54 = s*v53 + c*v54
            v03 = nv03; v04 = nv04; v13 = nv13; v14 = nv14; v23 = nv23; v24 = nv24; v33 = nv33; v34 = nv34; v43 = nv43; v44 = nv44; v53 = nv53; v54 = nv54
            # pair (3,5)
            off_diag = g35; diag_diff = g55 - g33; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng33 = c*c*g33 - 2.0*s*c*g35 + s*s*g55; ng55 = s*s*g33 + 2.0*s*c*g35 + c*c*g55
            ng03 = c*g03 - s*g05; ng05 = s*g03 + c*g05
            ng13 = c*g13 - s*g15; ng15 = s*g13 + c*g15
            ng23 = c*g23 - s*g25; ng25 = s*g23 + c*g25
            ng34 = c*g34 - s*g45; ng45 = s*g34 + c*g45
            g33 = ng33; g55 = ng55; g35 = tl.zeros([], dtype=DTYPE)
            g03 = ng03; g05 = ng05; g13 = ng13; g15 = ng15; g23 = ng23; g25 = ng25; g34 = ng34; g45 = ng45
            nv03 = c*v03 - s*v05; nv05 = s*v03 + c*v05
            nv13 = c*v13 - s*v15; nv15 = s*v13 + c*v15
            nv23 = c*v23 - s*v25; nv25 = s*v23 + c*v25
            nv33 = c*v33 - s*v35; nv35 = s*v33 + c*v35
            nv43 = c*v43 - s*v45; nv45 = s*v43 + c*v45
            nv53 = c*v53 - s*v55; nv55 = s*v53 + c*v55
            v03 = nv03; v05 = nv05; v13 = nv13; v15 = nv15; v23 = nv23; v25 = nv25; v33 = nv33; v35 = nv35; v43 = nv43; v45 = nv45; v53 = nv53; v55 = nv55
            # pair (4,5)
            off_diag = g45; diag_diff = g55 - g44; abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c
            ng44 = c*c*g44 - 2.0*s*c*g45 + s*s*g55; ng55 = s*s*g44 + 2.0*s*c*g45 + c*c*g55
            ng04 = c*g04 - s*g05; ng05 = s*g04 + c*g05
            ng14 = c*g14 - s*g15; ng15 = s*g14 + c*g15
            ng24 = c*g24 - s*g25; ng25 = s*g24 + c*g25
            ng34 = c*g34 - s*g35; ng35 = s*g34 + c*g35
            g44 = ng44; g55 = ng55; g45 = tl.zeros([], dtype=DTYPE)
            g04 = ng04; g05 = ng05; g14 = ng14; g15 = ng15; g24 = ng24; g25 = ng25; g34 = ng34; g35 = ng35
            nv04 = c*v04 - s*v05; nv05 = s*v04 + c*v05
            nv14 = c*v14 - s*v15; nv15 = s*v14 + c*v15
            nv24 = c*v24 - s*v25; nv25 = s*v24 + c*v25
            nv34 = c*v34 - s*v35; nv35 = s*v34 + c*v35
            nv44 = c*v44 - s*v45; nv45 = s*v44 + c*v45
            nv54 = c*v54 - s*v55; nv55 = s*v54 + c*v55
            v04 = nv04; v05 = nv05; v14 = nv14; v15 = nv15; v24 = nv24; v25 = nv25; v34 = nv34; v35 = nv35; v44 = nv44; v45 = nv45; v54 = nv54; v55 = nv55
        s0 = tl.sqrt(tl.maximum(g00, EPS))
        s1 = tl.sqrt(tl.maximum(g11, EPS))
        s2 = tl.sqrt(tl.maximum(g22, EPS))
        s3 = tl.sqrt(tl.maximum(g33, EPS))
        s4 = tl.sqrt(tl.maximum(g44, EPS))
        s5 = tl.sqrt(tl.maximum(g55, EPS))
        # Selection sort descending; permute V columns alongside.
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00; v00 = tl.where(do_swap, v01, v00); v01 = tl.where(do_swap, tv, v01)
        tv = v10; v10 = tl.where(do_swap, v11, v10); v11 = tl.where(do_swap, tv, v11)
        tv = v20; v20 = tl.where(do_swap, v21, v20); v21 = tl.where(do_swap, tv, v21)
        tv = v30; v30 = tl.where(do_swap, v31, v30); v31 = tl.where(do_swap, tv, v31)
        tv = v40; v40 = tl.where(do_swap, v41, v40); v41 = tl.where(do_swap, tv, v41)
        tv = v50; v50 = tl.where(do_swap, v51, v50); v51 = tl.where(do_swap, tv, v51)
        do_swap = s0 < s2
        s0, s2 = tl.where(do_swap, s2, s0), tl.where(do_swap, s0, s2)
        tv = v00; v00 = tl.where(do_swap, v02, v00); v02 = tl.where(do_swap, tv, v02)
        tv = v10; v10 = tl.where(do_swap, v12, v10); v12 = tl.where(do_swap, tv, v12)
        tv = v20; v20 = tl.where(do_swap, v22, v20); v22 = tl.where(do_swap, tv, v22)
        tv = v30; v30 = tl.where(do_swap, v32, v30); v32 = tl.where(do_swap, tv, v32)
        tv = v40; v40 = tl.where(do_swap, v42, v40); v42 = tl.where(do_swap, tv, v42)
        tv = v50; v50 = tl.where(do_swap, v52, v50); v52 = tl.where(do_swap, tv, v52)
        do_swap = s0 < s3
        s0, s3 = tl.where(do_swap, s3, s0), tl.where(do_swap, s0, s3)
        tv = v00; v00 = tl.where(do_swap, v03, v00); v03 = tl.where(do_swap, tv, v03)
        tv = v10; v10 = tl.where(do_swap, v13, v10); v13 = tl.where(do_swap, tv, v13)
        tv = v20; v20 = tl.where(do_swap, v23, v20); v23 = tl.where(do_swap, tv, v23)
        tv = v30; v30 = tl.where(do_swap, v33, v30); v33 = tl.where(do_swap, tv, v33)
        tv = v40; v40 = tl.where(do_swap, v43, v40); v43 = tl.where(do_swap, tv, v43)
        tv = v50; v50 = tl.where(do_swap, v53, v50); v53 = tl.where(do_swap, tv, v53)
        do_swap = s0 < s4
        s0, s4 = tl.where(do_swap, s4, s0), tl.where(do_swap, s0, s4)
        tv = v00; v00 = tl.where(do_swap, v04, v00); v04 = tl.where(do_swap, tv, v04)
        tv = v10; v10 = tl.where(do_swap, v14, v10); v14 = tl.where(do_swap, tv, v14)
        tv = v20; v20 = tl.where(do_swap, v24, v20); v24 = tl.where(do_swap, tv, v24)
        tv = v30; v30 = tl.where(do_swap, v34, v30); v34 = tl.where(do_swap, tv, v34)
        tv = v40; v40 = tl.where(do_swap, v44, v40); v44 = tl.where(do_swap, tv, v44)
        tv = v50; v50 = tl.where(do_swap, v54, v50); v54 = tl.where(do_swap, tv, v54)
        do_swap = s0 < s5
        s0, s5 = tl.where(do_swap, s5, s0), tl.where(do_swap, s0, s5)
        tv = v00; v00 = tl.where(do_swap, v05, v00); v05 = tl.where(do_swap, tv, v05)
        tv = v10; v10 = tl.where(do_swap, v15, v10); v15 = tl.where(do_swap, tv, v15)
        tv = v20; v20 = tl.where(do_swap, v25, v20); v25 = tl.where(do_swap, tv, v25)
        tv = v30; v30 = tl.where(do_swap, v35, v30); v35 = tl.where(do_swap, tv, v35)
        tv = v40; v40 = tl.where(do_swap, v45, v40); v45 = tl.where(do_swap, tv, v45)
        tv = v50; v50 = tl.where(do_swap, v55, v50); v55 = tl.where(do_swap, tv, v55)
        do_swap = s1 < s2
        s1, s2 = tl.where(do_swap, s2, s1), tl.where(do_swap, s1, s2)
        tv = v01; v01 = tl.where(do_swap, v02, v01); v02 = tl.where(do_swap, tv, v02)
        tv = v11; v11 = tl.where(do_swap, v12, v11); v12 = tl.where(do_swap, tv, v12)
        tv = v21; v21 = tl.where(do_swap, v22, v21); v22 = tl.where(do_swap, tv, v22)
        tv = v31; v31 = tl.where(do_swap, v32, v31); v32 = tl.where(do_swap, tv, v32)
        tv = v41; v41 = tl.where(do_swap, v42, v41); v42 = tl.where(do_swap, tv, v42)
        tv = v51; v51 = tl.where(do_swap, v52, v51); v52 = tl.where(do_swap, tv, v52)
        do_swap = s1 < s3
        s1, s3 = tl.where(do_swap, s3, s1), tl.where(do_swap, s1, s3)
        tv = v01; v01 = tl.where(do_swap, v03, v01); v03 = tl.where(do_swap, tv, v03)
        tv = v11; v11 = tl.where(do_swap, v13, v11); v13 = tl.where(do_swap, tv, v13)
        tv = v21; v21 = tl.where(do_swap, v23, v21); v23 = tl.where(do_swap, tv, v23)
        tv = v31; v31 = tl.where(do_swap, v33, v31); v33 = tl.where(do_swap, tv, v33)
        tv = v41; v41 = tl.where(do_swap, v43, v41); v43 = tl.where(do_swap, tv, v43)
        tv = v51; v51 = tl.where(do_swap, v53, v51); v53 = tl.where(do_swap, tv, v53)
        do_swap = s1 < s4
        s1, s4 = tl.where(do_swap, s4, s1), tl.where(do_swap, s1, s4)
        tv = v01; v01 = tl.where(do_swap, v04, v01); v04 = tl.where(do_swap, tv, v04)
        tv = v11; v11 = tl.where(do_swap, v14, v11); v14 = tl.where(do_swap, tv, v14)
        tv = v21; v21 = tl.where(do_swap, v24, v21); v24 = tl.where(do_swap, tv, v24)
        tv = v31; v31 = tl.where(do_swap, v34, v31); v34 = tl.where(do_swap, tv, v34)
        tv = v41; v41 = tl.where(do_swap, v44, v41); v44 = tl.where(do_swap, tv, v44)
        tv = v51; v51 = tl.where(do_swap, v54, v51); v54 = tl.where(do_swap, tv, v54)
        do_swap = s1 < s5
        s1, s5 = tl.where(do_swap, s5, s1), tl.where(do_swap, s1, s5)
        tv = v01; v01 = tl.where(do_swap, v05, v01); v05 = tl.where(do_swap, tv, v05)
        tv = v11; v11 = tl.where(do_swap, v15, v11); v15 = tl.where(do_swap, tv, v15)
        tv = v21; v21 = tl.where(do_swap, v25, v21); v25 = tl.where(do_swap, tv, v25)
        tv = v31; v31 = tl.where(do_swap, v35, v31); v35 = tl.where(do_swap, tv, v35)
        tv = v41; v41 = tl.where(do_swap, v45, v41); v45 = tl.where(do_swap, tv, v45)
        tv = v51; v51 = tl.where(do_swap, v55, v51); v55 = tl.where(do_swap, tv, v55)
        do_swap = s2 < s3
        s2, s3 = tl.where(do_swap, s3, s2), tl.where(do_swap, s2, s3)
        tv = v02; v02 = tl.where(do_swap, v03, v02); v03 = tl.where(do_swap, tv, v03)
        tv = v12; v12 = tl.where(do_swap, v13, v12); v13 = tl.where(do_swap, tv, v13)
        tv = v22; v22 = tl.where(do_swap, v23, v22); v23 = tl.where(do_swap, tv, v23)
        tv = v32; v32 = tl.where(do_swap, v33, v32); v33 = tl.where(do_swap, tv, v33)
        tv = v42; v42 = tl.where(do_swap, v43, v42); v43 = tl.where(do_swap, tv, v43)
        tv = v52; v52 = tl.where(do_swap, v53, v52); v53 = tl.where(do_swap, tv, v53)
        do_swap = s2 < s4
        s2, s4 = tl.where(do_swap, s4, s2), tl.where(do_swap, s2, s4)
        tv = v02; v02 = tl.where(do_swap, v04, v02); v04 = tl.where(do_swap, tv, v04)
        tv = v12; v12 = tl.where(do_swap, v14, v12); v14 = tl.where(do_swap, tv, v14)
        tv = v22; v22 = tl.where(do_swap, v24, v22); v24 = tl.where(do_swap, tv, v24)
        tv = v32; v32 = tl.where(do_swap, v34, v32); v34 = tl.where(do_swap, tv, v34)
        tv = v42; v42 = tl.where(do_swap, v44, v42); v44 = tl.where(do_swap, tv, v44)
        tv = v52; v52 = tl.where(do_swap, v54, v52); v54 = tl.where(do_swap, tv, v54)
        do_swap = s2 < s5
        s2, s5 = tl.where(do_swap, s5, s2), tl.where(do_swap, s2, s5)
        tv = v02; v02 = tl.where(do_swap, v05, v02); v05 = tl.where(do_swap, tv, v05)
        tv = v12; v12 = tl.where(do_swap, v15, v12); v15 = tl.where(do_swap, tv, v15)
        tv = v22; v22 = tl.where(do_swap, v25, v22); v25 = tl.where(do_swap, tv, v25)
        tv = v32; v32 = tl.where(do_swap, v35, v32); v35 = tl.where(do_swap, tv, v35)
        tv = v42; v42 = tl.where(do_swap, v45, v42); v45 = tl.where(do_swap, tv, v45)
        tv = v52; v52 = tl.where(do_swap, v55, v52); v55 = tl.where(do_swap, tv, v55)
        do_swap = s3 < s4
        s3, s4 = tl.where(do_swap, s4, s3), tl.where(do_swap, s3, s4)
        tv = v03; v03 = tl.where(do_swap, v04, v03); v04 = tl.where(do_swap, tv, v04)
        tv = v13; v13 = tl.where(do_swap, v14, v13); v14 = tl.where(do_swap, tv, v14)
        tv = v23; v23 = tl.where(do_swap, v24, v23); v24 = tl.where(do_swap, tv, v24)
        tv = v33; v33 = tl.where(do_swap, v34, v33); v34 = tl.where(do_swap, tv, v34)
        tv = v43; v43 = tl.where(do_swap, v44, v43); v44 = tl.where(do_swap, tv, v44)
        tv = v53; v53 = tl.where(do_swap, v54, v53); v54 = tl.where(do_swap, tv, v54)
        do_swap = s3 < s5
        s3, s5 = tl.where(do_swap, s5, s3), tl.where(do_swap, s3, s5)
        tv = v03; v03 = tl.where(do_swap, v05, v03); v05 = tl.where(do_swap, tv, v05)
        tv = v13; v13 = tl.where(do_swap, v15, v13); v15 = tl.where(do_swap, tv, v15)
        tv = v23; v23 = tl.where(do_swap, v25, v23); v25 = tl.where(do_swap, tv, v25)
        tv = v33; v33 = tl.where(do_swap, v35, v33); v35 = tl.where(do_swap, tv, v35)
        tv = v43; v43 = tl.where(do_swap, v45, v43); v45 = tl.where(do_swap, tv, v45)
        tv = v53; v53 = tl.where(do_swap, v55, v53); v55 = tl.where(do_swap, tv, v55)
        do_swap = s4 < s5
        s4, s5 = tl.where(do_swap, s5, s4), tl.where(do_swap, s4, s5)
        tv = v04; v04 = tl.where(do_swap, v05, v04); v05 = tl.where(do_swap, tv, v05)
        tv = v14; v14 = tl.where(do_swap, v15, v14); v15 = tl.where(do_swap, tv, v15)
        tv = v24; v24 = tl.where(do_swap, v25, v24); v25 = tl.where(do_swap, tv, v25)
        tv = v34; v34 = tl.where(do_swap, v35, v34); v35 = tl.where(do_swap, tv, v35)
        tv = v44; v44 = tl.where(do_swap, v45, v44); v45 = tl.where(do_swap, tv, v45)
        tv = v54; v54 = tl.where(do_swap, v55, v54); v55 = tl.where(do_swap, tv, v55)
        s_base = bid * 6
        tl.store(S_ptr + s_base + 0, s0)
        tl.store(S_ptr + s_base + 1, s1)
        tl.store(S_ptr + s_base + 2, s2)
        tl.store(S_ptr + s_base + 3, s3)
        tl.store(S_ptr + s_base + 4, s4)
        tl.store(S_ptr + s_base + 5, s5)
        vh_base = bid * 36
        tl.store(Vh_ptr + vh_base + 0, v00)
        tl.store(Vh_ptr + vh_base + 1, v10)
        tl.store(Vh_ptr + vh_base + 2, v20)
        tl.store(Vh_ptr + vh_base + 3, v30)
        tl.store(Vh_ptr + vh_base + 4, v40)
        tl.store(Vh_ptr + vh_base + 5, v50)
        tl.store(Vh_ptr + vh_base + 6, v01)
        tl.store(Vh_ptr + vh_base + 7, v11)
        tl.store(Vh_ptr + vh_base + 8, v21)
        tl.store(Vh_ptr + vh_base + 9, v31)
        tl.store(Vh_ptr + vh_base + 10, v41)
        tl.store(Vh_ptr + vh_base + 11, v51)
        tl.store(Vh_ptr + vh_base + 12, v02)
        tl.store(Vh_ptr + vh_base + 13, v12)
        tl.store(Vh_ptr + vh_base + 14, v22)
        tl.store(Vh_ptr + vh_base + 15, v32)
        tl.store(Vh_ptr + vh_base + 16, v42)
        tl.store(Vh_ptr + vh_base + 17, v52)
        tl.store(Vh_ptr + vh_base + 18, v03)
        tl.store(Vh_ptr + vh_base + 19, v13)
        tl.store(Vh_ptr + vh_base + 20, v23)
        tl.store(Vh_ptr + vh_base + 21, v33)
        tl.store(Vh_ptr + vh_base + 22, v43)
        tl.store(Vh_ptr + vh_base + 23, v53)
        tl.store(Vh_ptr + vh_base + 24, v04)
        tl.store(Vh_ptr + vh_base + 25, v14)
        tl.store(Vh_ptr + vh_base + 26, v24)
        tl.store(Vh_ptr + vh_base + 27, v34)
        tl.store(Vh_ptr + vh_base + 28, v44)
        tl.store(Vh_ptr + vh_base + 29, v54)
        tl.store(Vh_ptr + vh_base + 30, v05)
        tl.store(Vh_ptr + vh_base + 31, v15)
        tl.store(Vh_ptr + vh_base + 32, v25)
        tl.store(Vh_ptr + vh_base + 33, v35)
        tl.store(Vh_ptr + vh_base + 34, v45)
        tl.store(Vh_ptr + vh_base + 35, v55)
        inv_s0 = 1.0 / (s0 + EPS); inv_s1 = 1.0 / (s1 + EPS); inv_s2 = 1.0 / (s2 + EPS); inv_s3 = 1.0 / (s3 + EPS); inv_s4 = 1.0 / (s4 + EPS); inv_s5 = 1.0 / (s5 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 6 + 0, mask=mask, other=0.0).to(DTYPE)
            a1 = tl.load(A_ptr + base + row_idx * 6 + 1, mask=mask, other=0.0).to(DTYPE)
            a2 = tl.load(A_ptr + base + row_idx * 6 + 2, mask=mask, other=0.0).to(DTYPE)
            a3 = tl.load(A_ptr + base + row_idx * 6 + 3, mask=mask, other=0.0).to(DTYPE)
            a4 = tl.load(A_ptr + base + row_idx * 6 + 4, mask=mask, other=0.0).to(DTYPE)
            a5 = tl.load(A_ptr + base + row_idx * 6 + 5, mask=mask, other=0.0).to(DTYPE)
            u0 = (a0*v00 + a1*v10 + a2*v20 + a3*v30 + a4*v40 + a5*v50) * inv_s0
            u1 = (a0*v01 + a1*v11 + a2*v21 + a3*v31 + a4*v41 + a5*v51) * inv_s1
            u2 = (a0*v02 + a1*v12 + a2*v22 + a3*v32 + a4*v42 + a5*v52) * inv_s2
            u3 = (a0*v03 + a1*v13 + a2*v23 + a3*v33 + a4*v43 + a5*v53) * inv_s3
            u4 = (a0*v04 + a1*v14 + a2*v24 + a3*v34 + a4*v44 + a5*v54) * inv_s4
            u5 = (a0*v05 + a1*v15 + a2*v25 + a3*v35 + a4*v45 + a5*v55) * inv_s5
            u_base = bid * M * 6
            tl.store(U_ptr + u_base + row_idx * 6 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 6 + 1, u1, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 6 + 2, u2, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 6 + 3, u3, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 6 + 4, u4, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 6 + 5, u5, mask=mask)

    HAS_TRITON = True

except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _triton_dtype(A):
    """Map torch dtype → tl dtype for the kernel constexpr. Restricts to fp32 / fp64."""
    if A.dtype == torch.float32:
        return tl.float32
    if A.dtype == torch.float64:
        return tl.float64
    raise TypeError(f"Triton SVD kernels support fp32/fp64 only, got {A.dtype}")


def batched_svd2(A, block_m=128):
    """Fused Triton SVD for (B, M, 2) tensors. Falls back to torch if no Triton.

    Honors fp32 and fp64 input dtype; output dtype matches input.
    Returns: U (B,M,2), S (B,2), Vh (B,2,2)
    """
    if not HAS_TRITON or not A.is_cuda or A.dtype not in (torch.float32, torch.float64):
        return torch.linalg.svd(A, full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 2
    B, M, _ = A.shape
    A_c = A.contiguous()
    U = torch.empty((B, M, 2), dtype=A.dtype, device=A.device)
    S = torch.empty((B, 2), dtype=A.dtype, device=A.device)
    Vh = torch.empty((B, 2, 2), dtype=A.dtype, device=A.device)
    _svd2_kernel[(B,)](A_c, U, S, Vh, M=M, BLOCK_M=block_m,
                       DTYPE=_triton_dtype(A), EPS=1e-12)
    return U, S, Vh


def batched_svd3(A, block_m=128, jacobi_iters=6):
    """Fused Triton SVD for (B, M, 3) tensors. fp32/fp64.

    Returns: U (B,M,3), S (B,3), Vh (B,3,3)
    """
    if not HAS_TRITON or not A.is_cuda or A.dtype not in (torch.float32, torch.float64):
        return torch.linalg.svd(A, full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 3
    B, M, _ = A.shape
    A_c = A.contiguous()
    U = torch.empty((B, M, 3), dtype=A.dtype, device=A.device)
    S = torch.empty((B, 3), dtype=A.dtype, device=A.device)
    Vh = torch.empty((B, 3, 3), dtype=A.dtype, device=A.device)
    _svd3_kernel[(B,)](A_c, U, S, Vh, M=M, BLOCK_M=block_m,
                       JACOBI_ITERS=jacobi_iters,
                       DTYPE=_triton_dtype(A), EPS=1e-12)
    return U, S, Vh


def batched_svd4(A, block_m=128, jacobi_iters=6):
    """Fused Triton SVD for (B, M, 4) tensors. fp32/fp64.

    Returns: U (B,M,4), S (B,4), Vh (B,4,4)
    """
    if not HAS_TRITON or not A.is_cuda or A.dtype not in (torch.float32, torch.float64):
        return torch.linalg.svd(A, full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 4
    B, M, _ = A.shape
    A_c = A.contiguous()
    U = torch.empty((B, M, 4), dtype=A.dtype, device=A.device)
    S = torch.empty((B, 4), dtype=A.dtype, device=A.device)
    Vh = torch.empty((B, 4, 4), dtype=A.dtype, device=A.device)
    _svd4_kernel[(B,)](A_c, U, S, Vh, M=M, BLOCK_M=block_m,
                       JACOBI_ITERS=jacobi_iters,
                       DTYPE=_triton_dtype(A), EPS=1e-12)
    return U, S, Vh


def batched_svd5(A, block_m=128, jacobi_iters=6):
    """Fused Triton SVD for (B, M, 5) tensors. fp32/fp64.

    Returns: U (B,M,5), S (B,5), Vh (B,5,5)
    """
    if not HAS_TRITON or not A.is_cuda or A.dtype not in (torch.float32, torch.float64):
        return torch.linalg.svd(A, full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 5
    B, M, _ = A.shape
    A_c = A.contiguous()
    U = torch.empty((B, M, 5), dtype=A.dtype, device=A.device)
    S = torch.empty((B, 5), dtype=A.dtype, device=A.device)
    Vh = torch.empty((B, 5, 5), dtype=A.dtype, device=A.device)
    _svd5_kernel[(B,)](A_c, U, S, Vh, M=M, BLOCK_M=block_m,
                       JACOBI_ITERS=jacobi_iters,
                       DTYPE=_triton_dtype(A), EPS=1e-12)
    return U, S, Vh


def batched_svd6(A, block_m=128, jacobi_iters=12):
    """Fused Triton SVD for (B, M, 6) tensors. fp32/fp64.

    Default jacobi_iters=12: N=6 has 15 pairs/sweep; at fp32 the
    accumulated rotation error from 6 sweeps left U^T U with ~4e-3
    deviation on Blackwell (above the 1e-3 orth tolerance). 12 sweeps
    (180 rotations) puts orth error well under 1e-3 with negligible
    runtime cost vs. eigh fallback.

    Returns: U (B,M,6), S (B,6), Vh (B,6,6)
    """
    if not HAS_TRITON or not A.is_cuda or A.dtype not in (torch.float32, torch.float64):
        return torch.linalg.svd(A, full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 6
    B, M, _ = A.shape
    A_c = A.contiguous()
    U = torch.empty((B, M, 6), dtype=A.dtype, device=A.device)
    S = torch.empty((B, 6), dtype=A.dtype, device=A.device)
    Vh = torch.empty((B, 6, 6), dtype=A.dtype, device=A.device)
    _svd6_kernel[(B,)](A_c, U, S, Vh, M=M, BLOCK_M=block_m,
                       JACOBI_ITERS=jacobi_iters,
                       DTYPE=_triton_dtype(A), EPS=1e-12)
    return U, S, Vh


# ═══════════════════════════════════════════════════════════════════════════════
# GRAM-EIGH HYBRID (N ≥ 4)
# ═══════════════════════════════════════════════════════════════════════════════

def gram_eigh_svd(A):
    """Thin SVD via Gram matrix eigendecomposition. Works for any N.

    G = A^T A → eigh(G) → S = sqrt(eigenvalues), V = eigenvectors, U = AV/S

    AMP-safe: disables autocast internally to prevent bf16 eigh failure.

    Args:
        A: (B, M, N) tensor, M >= N

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    B, M, N = A.shape
    with torch.amp.autocast('cuda', enabled=False):
        A_f = A.float()
        G = torch.bmm(A_f.transpose(1, 2), A_f)
        eigenvalues, V = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-12))
        U = torch.bmm(A_f, V) / S.unsqueeze(1)
        Vh = V.transpose(-2, -1).contiguous()
    return U, S, Vh


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

_TRITON_SVD_WRAPPERS = {}  # filled below if HAS_TRITON


def batched_svd(A, method='auto', block_m=128):
    """Batched thin SVD for (B, M, N) tensors. M >= N.

    Auto-dispatches by N:
      N=2..6: Fused Triton (fp32 or fp64)
      N>=7:   Gram + eigh

    Note: very large N hits eigh serialization. For Procrustes alignment at
    large N, use batched_procrustes() which bypasses this.

    Args:
        A:        (B, M, N) tensor, CUDA for Triton kernels
        method:   'auto', 'triton', 'gram_eigh', 'torch'
        block_m:  Tile size for Triton kernels

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    assert A.ndim == 3, f"Expected (B, M, N), got {A.shape}"
    B, M, N = A.shape
    assert M >= N, f"Thin SVD requires M >= N, got M={M}, N={N}"

    triton_eligible = HAS_TRITON and A.is_cuda and A.dtype in (torch.float32, torch.float64)

    if method == 'auto':
        if 2 <= N <= 6 and triton_eligible:
            return _TRITON_SVD_WRAPPERS[N](A, block_m)
        return gram_eigh_svd(A)
    elif method == 'triton':
        if N not in (2, 3, 4, 5, 6):
            raise ValueError(f"Triton kernel only for N=2..6, got N={N}")
        return _TRITON_SVD_WRAPPERS[N](A, block_m)
    elif method == 'gram_eigh':
        return gram_eigh_svd(A)
    elif method == 'torch':
        return torch.linalg.svd(A, full_matrices=False)
    raise ValueError(f"Unknown method '{method}'. Use: auto, triton, gram_eigh, torch")


if HAS_TRITON:
    _TRITON_SVD_WRAPPERS = {
        2: batched_svd2,
        3: batched_svd3,
        4: batched_svd4,
        5: batched_svd5,
        6: batched_svd6,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NEWTON-SCHULZ INVERSE SQUARE ROOT
# ═══════════════════════════════════════════════════════════════════════════════

def newton_schulz_invsqrt(G, iters=10):
    """Batched G^{-1/2} via Newton-Schulz iteration.

    Pure bmm — zero eigensolvers. Quadratic convergence.
    Use for Procrustes whitening: W = X @ newton_schulz_invsqrt(X^T X)

    AMP-safe: disables autocast internally.

    Args:
        G:     (B, N, N) symmetric PSD matrices
        iters: Iteration count (10 conservative, 7 usually sufficient)

    Returns: (B, N, N) inverse square root matrices
    """
    B, N, _ = G.shape
    device = G.device
    with torch.amp.autocast('cuda', enabled=False):
        G = G.float()
        trace = G.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
        G_norm = G / trace
        I = torch.eye(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        Y = G_norm.clone()
        Z = I.clone()
        for _ in range(iters):
            ZY = torch.bmm(Z, Y)
            factor = 1.5 * I - 0.5 * ZY
            Y = torch.bmm(Y, factor)
            Z = torch.bmm(factor, Z)
        return Z / trace.sqrt()


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSPACE-PRESERVING PROCRUSTES ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def batched_procrustes(source, target, rank=24, whiten=True, schulz_iters=10):
    """Batched Procrustes alignment with subspace-preserving rotation.

    N ≤ 32: full N-d Procrustes via SVD (sub-ms).
    N > 32: project to rank-d, align there, lift back preserving
            orthogonal complement exactly.

    Validated: 1.000 nearest-neighbor agreement with full Procrustes
    across N=32-128, k=8-64.

    AMP-safe: disables autocast internally.

    Args:
        source:       (B, n_samples, N) or (n_samples, N) — source embeddings
        target:       (B, n_samples, N) or (n_samples, N) — target embeddings
        rank:         Projection rank for N > 32 (default 24)
        whiten:       Apply Newton-Schulz whitening (default True)
        schulz_iters: Iterations for whitening (default 10)

    Returns:
        aligned: same shape as source — source aligned to target
        info:    dict with method, rotation matrix, diagnostics
    """
    unbatched = source.ndim == 2
    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)

    B, n_samples, N = source.shape
    device = source.device

    with torch.amp.autocast('cuda', enabled=False):
        source_f = source.float()
        target_f = target.float()

        # Center
        src_mean = source_f.mean(1, keepdim=True)
        tgt_mean = target_f.mean(1, keepdim=True)
        src_c = source_f - src_mean
        tgt_c = target_f - tgt_mean

        # Whiten
        if whiten:
            src_cov = torch.bmm(src_c.transpose(1, 2), src_c) / max(n_samples - 1, 1)
            tgt_cov = torch.bmm(tgt_c.transpose(1, 2), tgt_c) / max(n_samples - 1, 1)
            src_W = newton_schulz_invsqrt(src_cov, iters=schulz_iters)
            tgt_W = newton_schulz_invsqrt(tgt_cov, iters=schulz_iters)
            src_w = F.normalize(torch.bmm(src_c, src_W), dim=-1)
            tgt_w = F.normalize(torch.bmm(tgt_c, tgt_W), dim=-1)
        else:
            src_w = src_c
            tgt_w = tgt_c

        use_projection = N > 32 and rank < N

        if not use_projection:
            # Full N-d Procrustes
            C = torch.bmm(src_w.transpose(1, 2), tgt_w)
            U, _, Vh = torch.linalg.svd(C)
            R = torch.bmm(U, Vh)
            aligned_w = torch.bmm(src_w, R)
            if whiten:
                aligned = torch.bmm(aligned_w, torch.linalg.pinv(tgt_W)) + tgt_mean
            else:
                aligned = aligned_w + tgt_mean
            cos_after = F.cosine_similarity(
                aligned_w[:, :min(1000, n_samples)],
                tgt_w[:, :min(1000, n_samples)], dim=-1).mean().item()
            info = {'method': 'full', 'N': N, 'rank': N,
                    'rotation': R, 'cos_after': cos_after}
        else:
            # Subspace-preserving rank-k Procrustes
            k = min(rank, N - 1)
            P = torch.linalg.qr(
                torch.randn(B, N, k, device=device, dtype=torch.float32)).Q
            src_proj = torch.bmm(src_w, P)
            tgt_proj = torch.bmm(tgt_w, P)
            C_k = torch.bmm(src_proj.transpose(1, 2), tgt_proj)
            U_k, _, Vh_k = torch.linalg.svd(C_k)
            R_k = torch.bmm(U_k, Vh_k)
            # Decompose and rotate only in-subspace
            src_in = torch.bmm(src_w, P)
            P_T = P.transpose(1, 2)
            src_perp = src_w - torch.bmm(src_in, P_T)
            src_rotated = torch.bmm(torch.bmm(src_in, R_k), P_T)
            aligned_w = src_rotated + src_perp
            if whiten:
                aligned = torch.bmm(aligned_w, torch.linalg.pinv(tgt_W)) + tgt_mean
            else:
                aligned = aligned_w + tgt_mean
            cos_after = F.cosine_similarity(
                aligned_w[:, :min(1000, n_samples)],
                tgt_w[:, :min(1000, n_samples)], dim=-1).mean().item()
            info = {'method': 'subspace', 'N': N, 'rank': k,
                    'rotation_k': R_k, 'projection': P, 'cos_after': cos_after}

    if unbatched:
        aligned = aligned.squeeze(0)

    return aligned, info


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"kernel.py — validation on {device}")
    print(f"  HAS_TRITON: {HAS_TRITON}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    B, M = 32, 256
    _counts = {'passed': 0, 'failed': 0}

    def _check(name, condition, detail=""):
        if condition:
            _counts['passed'] += 1
            print(f"  [PASS] {name}")
        else:
            _counts['failed'] += 1
            print(f"  [FAIL] {name}  {detail}")

    def _validate_svd(A, U, S, Vh, label):
        """Check reconstruction, orthogonality, descending S.

        Thin SVD: U is (B, M, K), S is (B, K), Vh is (B, K, N) where K = min(M, N).
        Orthogonality: U^T U = I_K (M >= N case) and Vh Vh^T = I_K (M < N case).
        """
        B, M, N = A.shape
        K = U.shape[-1]
        recon = torch.bmm(U * S.unsqueeze(1), Vh)
        recon_err = (A.float() - recon).pow(2).mean().sqrt().item()
        UtU = torch.bmm(U.transpose(1, 2), U)
        I_K = torch.eye(K, device=A.device, dtype=UtU.dtype).unsqueeze(0)
        orth_err = (UtU - I_K).pow(2).mean().sqrt().item()
        desc = (S[:, :-1] >= S[:, 1:] - 1e-5).all().item()
        _check(f"{label} recon",  recon_err < 1e-3, f"err={recon_err:.2e}")
        _check(f"{label} orth",   orth_err < 1e-3,  f"err={orth_err:.2e}")
        _check(f"{label} desc",   desc)

    # ── batched_svd auto-dispatch ──
    print("batched_svd (auto-dispatch):")
    for N in [2, 3, 8, 16, 32]:
        A = torch.randn(B, M, N, device=device)
        U, S, Vh = batched_svd(A)
        _check(f"  N={N:>2} shapes", U.shape == (B, M, N) and S.shape == (B, N) and Vh.shape == (B, N, N))
        _validate_svd(A, U, S, Vh, f"  N={N:>2}")

    # ── Triton kernels explicitly ──
    if HAS_TRITON and device == 'cuda':
        print("\nbatched_svd2 (Triton):")
        A2 = torch.randn(B, M, 2, device=device)
        U2, S2, Vh2 = batched_svd2(A2)
        _validate_svd(A2, U2, S2, Vh2, "  N=2 triton")

        print("\nbatched_svd3 (Triton):")
        A3 = torch.randn(B, M, 3, device=device)
        U3, S3, Vh3 = batched_svd3(A3)
        _validate_svd(A3, U3, S3, Vh3, "  N=3 triton")

    # ── gram_eigh_svd directly ──
    print("\ngram_eigh_svd:")
    for N in [4, 24, 48]:
        A = torch.randn(B, M, N, device=device)
        U, S, Vh = gram_eigh_svd(A)
        _validate_svd(A, U, S, Vh, f"  N={N}")

    # ── newton_schulz_invsqrt ──
    print("\nnewton_schulz_invsqrt:")
    N = 16
    X = torch.randn(B, 100, N, device=device)
    G = torch.bmm(X.transpose(1, 2), X) / 99
    G_inv_sqrt = newton_schulz_invsqrt(G)
    # G_inv_sqrt @ G @ G_inv_sqrt should ≈ I
    product = torch.bmm(torch.bmm(G_inv_sqrt, G), G_inv_sqrt)
    I_N = torch.eye(N, device=device).unsqueeze(0)
    ns_err = (product - I_N).pow(2).mean().sqrt().item()
    _check("  invsqrt identity", ns_err < 1e-2, f"err={ns_err:.2e}")
    _check("  invsqrt shape", G_inv_sqrt.shape == (B, N, N))

    # ── batched_procrustes (full, N ≤ 32) ──
    # Use a real rotation between src and tgt so Procrustes has something to
    # recover. Without a rotation, whitening + F.normalize destroy magnitudes
    # and the "improved" check becomes RNG-dependent (was the prior failure
    # mode). Seed locally for reproducibility.
    print("\nbatched_procrustes (full):")
    _g = torch.Generator(device=device).manual_seed(20260501)
    N = 24
    src = torch.randn(500, N, device=device, generator=_g)
    # Random orthogonal R via QR; tgt = src @ R + small noise.
    Q, _ = torch.linalg.qr(torch.randn(N, N, device=device, generator=_g))
    tgt = src @ Q + 0.05 * torch.randn(500, N, device=device, generator=_g)
    cos_before = F.cosine_similarity(src, tgt, dim=-1).mean().item()
    aligned, info = batched_procrustes(src, tgt, rank=24)
    cos_after = F.cosine_similarity(aligned, tgt, dim=-1).mean().item()
    _check("  full method",  info['method'] == 'full')
    _check("  full shape",   aligned.shape == src.shape)
    _check("  full improved", cos_after > cos_before, f"{cos_before:.4f} → {cos_after:.4f}")

    # ── batched_procrustes (subspace, N > 32) ──
    print("\nbatched_procrustes (subspace):")
    _g = torch.Generator(device=device).manual_seed(20260502)
    N = 64
    src = torch.randn(500, N, device=device, generator=_g)
    Q, _ = torch.linalg.qr(torch.randn(N, N, device=device, generator=_g))
    tgt = src @ Q + 0.05 * torch.randn(500, N, device=device, generator=_g)
    cos_before = F.cosine_similarity(src, tgt, dim=-1).mean().item()
    aligned, info = batched_procrustes(src, tgt, rank=24)
    cos_after = F.cosine_similarity(aligned, tgt, dim=-1).mean().item()
    _check("  subspace method",  info['method'] == 'subspace')
    _check("  subspace rank",    info['rank'] == 24)
    _check("  subspace shape",   aligned.shape == src.shape)
    _check("  subspace improved", cos_after > cos_before, f"{cos_before:.4f} → {cos_after:.4f}")

    # ── batched interface ──
    print("\nbatched_procrustes (batched):")
    src_b = torch.randn(4, 200, 32, device=device)
    tgt_b = src_b * 0.5 + torch.randn_like(src_b) * 0.3
    aligned_b, info_b = batched_procrustes(src_b, tgt_b)
    _check("  batched shape", aligned_b.shape == src_b.shape)
    _check("  batched method", info_b['method'] == 'full')

    # ── SVD Triton size-sweep battery ─────────────────────────────────
    # Forced visibility: the previous battery section ran silently when an
    # older site-packages copy of geolip_core was loaded ahead of the worktree.
    # Print + flush a banner so it's obvious whether this section is reached,
    # report the loaded module path so stale installs are caught, and wrap the
    # body in try/except so any error reaches stdout (not just stderr).
    import sys as _sys
    print("\n" + "="*50, flush=True)
    print("SVD Triton size-sweep battery", flush=True)
    print("="*50, flush=True)
    print(f"  loaded from: {__file__}", flush=True)
    print(f"  python: {_sys.version.split()[0]}", flush=True)

    try:
        from geolip_core.linalg._backend import backend as _be
        # NB: `geolip_core.linalg.__init__` rebinds `svd = batched_svd`, so
        # `from geolip_core.linalg import svd` returns the FUNCTION, not the
        # submodule. Import the function from the submodule directly.
        from geolip_core.linalg.svd import batched_svd as _LA_batched_svd
        import geolip_core as _gc
        print(f"  geolip_core package: {getattr(_gc, '__file__', '?')}", flush=True)
        _be.status()

        SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3),
                  (4, 2), (4, 3), (4, 4),
                  (5, 2), (5, 3), (5, 4), (5, 5),
                  (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
        B_T, M_OUTER = 32, 1024

        # 1) Shape × dtype sweep through linalg.svd(method='auto')
        for cdt in ('fp32', 'fp64'):
            torch_dt = torch.float32 if cdt == 'fp32' else torch.float64
            print(f"\n[auto / {cdt}]", flush=True)
            for (m, n) in SHAPES:
                # Tiny matrix, plus a stress run with M_OUTER rows for tall shapes.
                M_used_set = (m, M_OUTER) if m >= n else (m,)
                for M_used in M_used_set:
                    A = torch.randn(B_T, M_used, n, device=device, dtype=torch_dt)
                    U, S, Vh = _LA_batched_svd(A, method='auto', compute_dtype=cdt)
                    _check(f"  {cdt} {M_used}x{n} dtype",
                           U.dtype == torch_dt and S.dtype == torch_dt and Vh.dtype == torch_dt)
                    _validate_svd(A, U, S, Vh, f"  {cdt} {M_used}x{n}")

        # 2) Jacobi convergence sweep — direct backend access
        print("\n[convergence sweep — backend.resolve_svd_nN]", flush=True)
        if _be.use_triton and device == 'cuda':
            for cdt, torch_dt in (('fp32', torch.float32), ('fp64', torch.float64)):
                for n, resolver in ((3, _be.resolve_svd_n3),
                                    (4, _be.resolve_svd_n4),
                                    (5, _be.resolve_svd_n5),
                                    (6, _be.resolve_svd_n6)):
                    A = torch.randn(B_T, M_OUTER, n, device=device, dtype=torch_dt)
                    worst = []
                    for it in (2, 4, 6, 8, 12):
                        U, S, Vh = resolver(A, 128, it)
                        recon = torch.bmm(U * S.unsqueeze(1), Vh)
                        err = (A - recon).pow(2).mean().sqrt().item()
                        worst.append((it, err))
                    line = f"  {cdt} N={n}: " + " ".join(f"it={it}:{e:.2e}" for it, e in worst)
                    print(line, flush=True)
                    # Final iter count (12) must be at least as accurate as iter=2.
                    _check(f"  {cdt} N={n} convergence monotone",
                           worst[-1][1] <= worst[0][1] + 1e-6,
                           f"first={worst[0][1]:.2e} last={worst[-1][1]:.2e}")
        else:
            print("  (skipped — Triton disabled or CPU device)", flush=True)

        # 3) Backend toggle: triton-off must still pass via torch.linalg.svd fallback
        print("\n[backend toggle — Triton OFF]", flush=True)
        saved_triton = _be.use_triton
        _be.use_triton = False
        try:
            for cdt in ('fp32', 'fp64'):
                torch_dt = torch.float32 if cdt == 'fp32' else torch.float64
                for (m, n) in SHAPES:
                    A = torch.randn(B_T, m, n, device=device, dtype=torch_dt)
                    U, S, Vh = _LA_batched_svd(A, method='auto', compute_dtype=cdt)
                    _validate_svd(A, U, S, Vh, f"  off/{cdt} {m}x{n}")
        finally:
            _be.use_triton = saved_triton

        # 4) Throughput benchmark — mirror geolip_core/utils/triton/fl_eigh_gen.py
        # Pattern: warmup w iters, time r iters, report per-iter mean. Informational
        # only (no _check calls) — kernels already passed correctness above.
        if device == 'cuda' and _be.use_triton:
            import time as _time
            def _sync(): torch.cuda.synchronize()
            def _gt(fn, w=20, r=200):
                for _ in range(w):
                    fn()
                _sync(); _t0 = _time.perf_counter()
                for _ in range(r):
                    fn()
                _sync()
                return (_time.perf_counter() - _t0) / r
            def _fmt(s):
                if s < 1e-3: return f"{s*1e6:.1f}us"
                if s < 1:    return f"{s*1e3:.2f}ms"
                return f"{s:.3f}s"

            BENCH_B, BENCH_M = 512, 1024

            for cdt in ('fp32', 'fp64'):
                torch_dt = torch.float32 if cdt == 'fp32' else torch.float64
                print(f"\n[bench / {cdt}]  B={BENCH_B} M={BENCH_M}", flush=True)
                print(f"  {'shape':>10} | {'cuSOLVER':>10} | {'auto':>10} | "
                      f"{'triton':>10} | {'speedup':>8}", flush=True)
                print("  " + "-" * 64, flush=True)
                for n in (2, 3, 4, 5, 6):
                    A = torch.randn(BENCH_B, BENCH_M, n, device=device, dtype=torch_dt)
                    t_torch = _gt(lambda A=A: torch.linalg.svd(A, full_matrices=False))
                    t_auto  = _gt(lambda A=A, c=cdt: _LA_batched_svd(A, method='auto',  compute_dtype=c))
                    t_trit  = _gt(lambda A=A, c=cdt: _LA_batched_svd(A, method='triton', compute_dtype=c))
                    sp = t_torch / t_trit if t_trit > 0 else float('inf')
                    print(f"  {BENCH_M}x{n:<2d} | {_fmt(t_torch):>10} | {_fmt(t_auto):>10} | "
                          f"{_fmt(t_trit):>10} | {sp:>6.1f}x", flush=True)
                    del A
                torch.cuda.empty_cache()

            # Batch scaling at the worst case (M=1024, N=6, fp32) — mirrors the
            # FL eigh batch scaling table in fl_eigh_gen.py.
            print(f"\n[bench / batch scaling — fp32, M=1024, N=6]", flush=True)
            print(f"  {'B':>6} | {'cuSOLVER':>10} | {'triton':>10} | {'speedup':>8}", flush=True)
            print("  " + "-" * 44, flush=True)
            for Bx in (256, 512, 1024, 2048, 4096, 8192):
                Ax = torch.randn(Bx, 1024, 6, device=device, dtype=torch.float32)
                t1 = _gt(lambda Ax=Ax: torch.linalg.svd(Ax, full_matrices=False), 10, 50)
                t2 = _gt(lambda Ax=Ax: _LA_batched_svd(Ax, method='triton', compute_dtype='fp32'), 10, 50)
                sp = t1 / t2 if t2 > 0 else float('inf')
                print(f"  {Bx:>6} | {_fmt(t1):>10} | {_fmt(t2):>10} | {sp:>6.1f}x", flush=True)
                del Ax
                torch.cuda.empty_cache()
        else:
            print("\n[bench]  skipped — Triton disabled or CPU device", flush=True)

    except Exception as _battery_exc:
        # Surface failures on stdout so they appear in line with the other test
        # output instead of getting separated onto stderr.
        import traceback as _tb
        print("\n[battery FAULT] " + repr(_battery_exc), flush=True)
        _tb.print_exc(file=_sys.stdout)
        _sys.stdout.flush()
        _counts['failed'] += 1

    # ── Summary ──
    total = _counts['passed'] + _counts['failed']
    print(f"\n{'='*50}")
    print(f"  {_counts['passed']}/{total} passed" + (f"  ({_counts['failed']} FAILED)" if _counts['failed'] else "  — all clear"))
    print(f"{'='*50}")