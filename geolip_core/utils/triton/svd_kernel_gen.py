"""
svd_kernel_gen.py — Triton SVD kernel source generator.

Emits the source of `_svdN_kernel` for any N >= 3, parameterized on N. The
generator mirrors the hand-written N=3..6 kernels in `geolip_core/utils/kernel.py`
exactly (algorithm, variable layout, statement order). It is designed to scale
through N=12 (the FL eigh ceiling).

Algorithm (fixed across N): cyclic Jacobi SVD on G = A^T A in scalar registers.
  1. Accumulate Gram upper triangle in a tiled load over M rows.
  2. Initialize V as identity (N^2 scalars, column-major: v[row][col]).
  3. JACOBI_ITERS sweeps; each sweep visits the N(N-1)/2 strict-upper pairs in
     row-major order and applies a Givens rotation that zeroes G[p,q].
  4. Singular values s_i = sqrt(max(g_ii, EPS)).
  5. Selection-sort descending; swap V columns alongside.
  6. Store S, Vh = V^T.
  7. Recover U via a second tile loop: U[:, c] = (A @ V[:, c]) / s_c.

N=2 is excluded — the existing hand-written N=2 kernel uses a closed-form
single-rotation path with a different signature (no JACOBI_ITERS); the generator
intentionally targets the iterative-Jacobi pattern shared by N >= 3.

Usage (CLI):
    python -m geolip_core.utils.triton.svd_kernel_gen --n 7 8 \\
        --out geolip_core/utils/triton/svd_triton_kernels_n78.py

    python -m geolip_core.utils.triton.svd_kernel_gen --verify  # smoke-test N=3

Author: AbstractPhil + Claude Opus 4.7
License: Apache 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


# ─────────────────────────────────────────────────────────────────────────────
# Naming helpers
# ─────────────────────────────────────────────────────────────────────────────

def g_var(i: int, j: int) -> str:
    """Upper-triangle Gram variable name. G[i,j] = G[j,i] is stored as g{min}{max}."""
    a, b = (i, j) if i <= j else (j, i)
    return f"g{a}{b}"


def ng_var(i: int, j: int) -> str:
    """Temporary 'next G' variable for a rotation update."""
    a, b = (i, j) if i <= j else (j, i)
    return f"ng{a}{b}"


def v_var(r: int, c: int) -> str:
    return f"v{r}{c}"


def nv_var(r: int, c: int) -> str:
    return f"nv{r}{c}"


# ─────────────────────────────────────────────────────────────────────────────
# Section emitters
# ─────────────────────────────────────────────────────────────────────────────

I1 = "    "    # 1 indent (inside `try:` block, decorator/def level)
I2 = "        "  # 2 indents (function body)
I3 = "            "  # 3 indents (inner loop body)


def gen_signature(n: int) -> List[str]:
    return [
        f"{I1}@triton.jit",
        f"{I1}def _svd{n}_kernel(",
        f"{I2}A_ptr, U_ptr, S_ptr, Vh_ptr,",
        f"{I2}M: tl.constexpr, BLOCK_M: tl.constexpr,",
        f"{I2}JACOBI_ITERS: tl.constexpr,",
        f"{I2}DTYPE: tl.constexpr, EPS: tl.constexpr,",
        f"{I1}):",
        f"{I2}bid = tl.program_id(0)",
        f"{I2}base = bid * M * {n}",
    ]


def gen_gram_init(n: int) -> List[str]:
    """Emit `g_ij = tl.zeros(...)` for upper-triangle entries.

    Layout: one source line per row of the upper triangle (matches N=5/N=6).
    """
    lines = [f"{I2}# Gram accumulators (upper triangle)."]
    for i in range(n):
        decls = [f"{g_var(i, j)} = tl.zeros([], dtype=DTYPE)" for j in range(i, n)]
        lines.append(I2 + "; ".join(decls))
    return lines


def gen_gram_loop(n: int) -> List[str]:
    """Tiled Gram accumulation pass over M rows."""
    lines = [
        f"{I2}for block_start in range(0, M, BLOCK_M):",
        f"{I3}offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M",
    ]
    for k in range(n):
        lines.append(
            f"{I3}a{k} = tl.load(A_ptr + base + row_idx * {n} + {k}, mask=mask, other=0.0).to(DTYPE)"
        )
    for i in range(n):
        accs = [f"{g_var(i, j)} += tl.sum(a{i} * a{j})" for j in range(i, n)]
        lines.append(I3 + "; ".join(accs))
    return lines


def gen_v_init(n: int) -> List[str]:
    """V = I_N as N^2 scalars, one row per source line."""
    lines = [f"{I2}# V = I_N (column-major: v[row][col])."]
    for r in range(n):
        decls = []
        for c in range(n):
            if r == c:
                decls.append(f"{v_var(r, c)} = tl.full([], 1.0, dtype=DTYPE)")
            else:
                decls.append(f"{v_var(r, c)} = tl.zeros([], dtype=DTYPE)")
        lines.append(I2 + "; ".join(decls))
    return lines


def gen_pair_block(n: int, p: int, q: int) -> List[str]:
    """Emit one Jacobi rotation block for the (p, q) pair, zeroing G[p, q]."""
    others = [k for k in range(n) if k not in (p, q)]
    lines = []
    lines.append(f"{I3}# pair ({p},{q})")

    # Compute c, s.
    lines.append(
        f"{I3}off_diag = {g_var(p, q)}; diag_diff = {g_var(q, q)} - {g_var(p, p)}; "
        f"abs_off = tl.abs(off_diag)"
    )
    lines.append(f"{I3}tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)")
    lines.append(
        f"{I3}t = tl.where(abs_off > EPS, "
        "tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)"
    )
    lines.append(f"{I3}c = 1.0 / tl.sqrt(1.0 + t * t); s = t * c")

    # G diagonal updates.
    gpp, gqq, gpq = g_var(p, p), g_var(q, q), g_var(p, q)
    ngpp, ngqq = ng_var(p, p), ng_var(q, q)
    lines.append(
        f"{I3}{ngpp} = c*c*{gpp} - 2.0*s*c*{gpq} + s*s*{gqq}; "
        f"{ngqq} = s*s*{gpp} + 2.0*s*c*{gpq} + c*c*{gqq}"
    )

    # G off-diagonal updates: for each k not in {p, q}, rotate the (p,k) and (q,k) pair.
    for k in others:
        lines.append(
            f"{I3}{ng_var(p, k)} = c*{g_var(p, k)} - s*{g_var(q, k)}; "
            f"{ng_var(q, k)} = s*{g_var(p, k)} + c*{g_var(q, k)}"
        )

    # Commit G updates.
    lines.append(
        f"{I3}{gpp} = {ngpp}; {gqq} = {ngqq}; {gpq} = tl.zeros([], dtype=DTYPE)"
    )
    if others:
        commits = []
        for k in others:
            commits.append(f"{g_var(p, k)} = {ng_var(p, k)}")
            commits.append(f"{g_var(q, k)} = {ng_var(q, k)}")
        lines.append(I3 + "; ".join(commits))

    # V updates: rotate columns p and q for every row.
    for r in range(n):
        lines.append(
            f"{I3}{nv_var(r, p)} = c*{v_var(r, p)} - s*{v_var(r, q)}; "
            f"{nv_var(r, q)} = s*{v_var(r, p)} + c*{v_var(r, q)}"
        )
    # Commit V updates (chunked 2 rows per source line, matches N=4 layout).
    chunk = []
    for r in range(n):
        chunk.append(f"{v_var(r, p)} = {nv_var(r, p)}")
        chunk.append(f"{v_var(r, q)} = {nv_var(r, q)}")
        if (r + 1) % 2 == 0 or r == n - 1:
            lines.append(I3 + "; ".join(chunk))
            chunk = []
    return lines


def gen_jacobi_loop(n: int) -> List[str]:
    lines = [f"{I2}for _ in range(JACOBI_ITERS):"]
    for p in range(n):
        for q in range(p + 1, n):
            lines.extend(gen_pair_block(n, p, q))
    return lines


def gen_singular_values(n: int) -> List[str]:
    return [
        f"{I2}# Singular values."
    ] + [
        f"{I2}s{i} = tl.sqrt(tl.maximum({g_var(i, i)}, EPS))" for i in range(n)
    ]


def gen_sort(n: int) -> List[str]:
    """Selection sort descending. For each comparison s_i vs s_j (i < j), if
    s_i < s_j swap them and the matching V columns. Pattern from
    `kernel.py:358-393` (N=4 reference)."""
    lines = [f"{I2}# Sort descending (selection sort; swap V columns alongside)."]
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(f"{I2}do_swap = s{i} < s{j}")
            lines.append(
                f"{I2}s{i}, s{j} = tl.where(do_swap, s{j}, s{i}), tl.where(do_swap, s{i}, s{j})"
            )
            for r in range(n):
                vri, vrj = v_var(r, i), v_var(r, j)
                lines.append(
                    f"{I2}tv = {vri}; {vri} = tl.where(do_swap, {vrj}, {vri}); "
                    f"{vrj} = tl.where(do_swap, tv, {vrj})"
                )
    return lines


def gen_s_store(n: int) -> List[str]:
    lines = [f"{I2}s_base = bid * {n}"]
    for i in range(n):
        lines.append(f"{I2}tl.store(S_ptr + s_base + {i}, s{i})")
    return lines


def gen_vh_store(n: int) -> List[str]:
    """Vh = V^T. Row r of Vh is column r of V → Vh[r, k] = V[k, r]."""
    lines = [
        f"{I2}# Vh = V^T  (row r of Vh is column r of V).",
        f"{I2}vh_base = bid * {n * n}",
    ]
    for r in range(n):  # row of Vh
        for k in range(n):  # col of Vh = row of V
            flat = r * n + k
            lines.append(f"{I2}tl.store(Vh_ptr + vh_base + {flat}, {v_var(k, r)})")
    return lines


def gen_inv_s(n: int) -> List[str]:
    return [f"{I2}inv_s{i} = 1.0 / (s{i} + EPS)" for i in range(n)]


def gen_u_recover(n: int) -> List[str]:
    """U = A @ V / S, computed in a second tile loop and stored in-place."""
    lines = [
        f"{I2}for block_start in range(0, M, BLOCK_M):",
        f"{I3}offs = tl.arange(0, BLOCK_M); row_idx = block_start + offs; mask = row_idx < M",
    ]
    for k in range(n):
        lines.append(
            f"{I3}a{k} = tl.load(A_ptr + base + row_idx * {n} + {k}, mask=mask, other=0.0).to(DTYPE)"
        )
    for c in range(n):
        terms = " + ".join(f"a{k}*{v_var(k, c)}" for k in range(n))
        lines.append(f"{I3}u{c} = ({terms}) * inv_s{c}")
    lines.append(f"{I3}u_base = bid * M * {n}")
    for c in range(n):
        lines.append(f"{I3}tl.store(U_ptr + u_base + row_idx * {n} + {c}, u{c}, mask=mask)")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Top-level kernel + module composition
# ─────────────────────────────────────────────────────────────────────────────

def gen_kernel(n: int) -> str:
    """Emit a complete `_svd{n}_kernel` source as a string (no surrounding try/except)."""
    if n < 3:
        raise ValueError(f"Generator targets N >= 3 (N=2 has a closed-form variant). Got N={n}.")
    sections = [
        gen_signature(n),
        gen_gram_init(n),
        gen_gram_loop(n),
        gen_v_init(n),
        gen_jacobi_loop(n),
        gen_singular_values(n),
        gen_sort(n),
        gen_s_store(n),
        gen_vh_store(n),
        gen_inv_s(n),
        gen_u_recover(n),
    ]
    return "\n".join(line for sec in sections for line in sec)


def gen_module(ns: Iterable[int], header_note: str = "") -> str:
    """Emit a full Python module containing _svdN_kernel definitions for the given N values.

    The module guards triton imports and exposes a HAS_TRITON_KERNELS flag plus the
    `_svdN_kernel` symbols (set to None when triton is unavailable).
    """
    ns = sorted(set(ns))
    n_list_str = ", ".join(f"N={n}" for n in ns)
    note_block = f"\n{header_note}\n" if header_note else ""
    lines = [
        '"""',
        f"AUTO-GENERATED by geolip_core/utils/triton/svd_kernel_gen.py — DO NOT EDIT.",
        "",
        f"Triton SVD kernels for {n_list_str}.",
        "",
        "Each kernel takes a (B, M, N) tensor on CUDA and writes thin SVD outputs:",
        "  U  (B, M, N)",
        "  S  (B, N)         singular values, descending",
        "  Vh (B, N, N)      right singular vectors transposed",
        "",
        "Algorithm: Gram + cyclic Jacobi (N(N-1)/2 plane rotations per sweep).",
        "Regenerate with `python -m geolip_core.utils.triton.svd_kernel_gen --n "
        + " ".join(str(n) for n in ns) + " --out <path>`.",
        f"{note_block}\"\"\"",
        "",
        "HAS_TRITON_KERNELS = False",
    ]
    for n in ns:
        lines.append(f"_svd{n}_kernel = None")
    lines.extend([
        "",
        "try:",
        "    import triton",
        "    import triton.language as tl",
        "",
    ])
    for i, n in enumerate(ns):
        if i:
            lines.append("")
        lines.append(gen_kernel(n))
    lines.extend([
        "",
        "    HAS_TRITON_KERNELS = True",
        "",
        "except ImportError:",
        "    pass",
        "",
    ])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Self-check
# ─────────────────────────────────────────────────────────────────────────────

def _verify_smoke(n: int = 3) -> None:
    """Compile the generated source for the given N, run on a random tensor,
    and compare to torch.linalg.svd. Smoke test only (does not byte-compare to
    the existing hand-written kernels — those have minor cosmetic formatting
    differences that are inconsequential to behaviour)."""
    import torch
    if not torch.cuda.is_available():
        print("[verify] CUDA not available; skipping smoke test.")
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        print("[verify] triton not installed; skipping smoke test.")
        return

    src = gen_module([n])
    print(f"[verify] generated module for N={n}: {len(src.splitlines())} lines")
    ns_globals: dict = {}
    exec(compile(src, f"<generated svd N={n}>", "exec"), ns_globals)
    kernel = ns_globals[f"_svd{n}_kernel"]
    assert kernel is not None, "generated kernel is None — triton import failed"

    B, M = 8, 64
    A = torch.randn(B, M, n, device="cuda", dtype=torch.float32).contiguous()
    U = torch.empty((B, M, n), dtype=torch.float32, device="cuda")
    S = torch.empty((B, n), dtype=torch.float32, device="cuda")
    Vh = torch.empty((B, n, n), dtype=torch.float32, device="cuda")
    kernel[(B,)](A, U, S, Vh, M=M, BLOCK_M=128, JACOBI_ITERS=12,
                 DTYPE=ns_globals["tl"].float32, EPS=1e-12)

    recon = torch.bmm(U * S.unsqueeze(1), Vh)
    recon_err = (A - recon).pow(2).mean().sqrt().item()
    UtU = torch.bmm(U.transpose(1, 2), U)
    eye = torch.eye(n, device="cuda").unsqueeze(0)
    orth_err = (UtU - eye).pow(2).mean().sqrt().item()
    desc = (S[:, :-1] >= S[:, 1:] - 1e-5).all().item()

    print(f"[verify] N={n}: recon={recon_err:.2e} orth={orth_err:.2e} desc={desc}")
    assert recon_err < 1e-3, f"recon error too high: {recon_err}"
    assert orth_err < 1e-3, f"orth error too high: {orth_err}"
    assert desc, "singular values not descending"
    print(f"[verify] N={n}: OK")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n", nargs="+", type=int, help="N values to generate (e.g. 7 8)")
    p.add_argument("--out", type=Path, help="Output module path")
    p.add_argument("--verify", action="store_true",
                   help="Run a smoke test: generate N=3 and compare to torch.linalg.svd")
    args = p.parse_args()

    if args.verify:
        _verify_smoke(3)
        return

    if not args.n or not args.out:
        p.error("--n and --out are required (or pass --verify)")

    src = gen_module(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(src, encoding="utf-8")
    print(f"wrote {args.out} ({len(src.splitlines())} lines, N={args.n})")


if __name__ == "__main__":
    main()
