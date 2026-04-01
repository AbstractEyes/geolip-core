"""
FL Eigh Triton — Generated Kernel.
Python generates fully-unrolled Triton source with explicit named variables.
No lists, no 2D tensors, no tl.where assignment. Just loads, FMAs, stores.
"""
import math, time, gc, sys
import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def _gen_kernel(N):
    """Generate Triton kernel source for NxN eigendecomposition."""
    N2 = N * N
    L = []  # lines
    def E(s): L.append(s)
    def var(prefix, i, j=None):
        return f"{prefix}{i}" if j is None else f"{prefix}{i}_{j}"

    E("@triton.jit")
    E(f"def _fl_eigh_gen(A_ptr, evals_ptr, evecs_ptr, B, BLOCK_B: tl.constexpr):")
    E(f"    pid = tl.program_id(0)")
    E(f"    bid = pid * BLOCK_B + tl.arange(0, BLOCK_B)")
    E(f"    mask = bid < B")
    E(f"    off = bid * {N2}")

    # Load A
    E(f"    # Load A")
    E(f"    frob = tl.zeros((BLOCK_B,), dtype=tl.float64)")
    for i in range(N):
        for j in range(N):
            v = f"a{i}_{j}"
            E(f"    {v} = tl.load(A_ptr + off + {i*N+j}, mask=mask, other=0.0).to(tl.float64)")
            E(f"    frob = frob + {v} * {v}")

    # Pre-scale
    E(f"    sc = tl.sqrt(frob * {1.0/N})")
    E(f"    sc = tl.where(sc > 1e-12, sc, 1e-12)")
    E(f"    inv_sc = 1.0 / sc")
    for i in range(N):
        for j in range(N):
            E(f"    a{i}_{j} = a{i}_{j} * inv_sc")

    # Phase 1: FL coefficients only (dont store M)
    E(f"    # Phase 1: FL coefficients")
    for k in range(N+1):
        E(f"    c{k} = tl.zeros((BLOCK_B,), dtype=tl.float64)" +
          (f" + 1.0" if k == N else ""))

    # M starts as zero
    for i in range(N):
        for j in range(N):
            E(f"    m{i}_{j} = tl.zeros((BLOCK_B,), dtype=tl.float64)")

    for kk in range(1, N+1):
        E(f"    # FL iter k={kk}")
        c_diag = f"c{N - kk + 1}"
        # mn = A @ m + c_diag * I
        for i in range(N):
            for j in range(N):
                terms = [f"a{i}_{l} * m{l}_{j}" for l in range(N)]
                expr = " + ".join(terms)
                if i == j:
                    expr += f" + {c_diag}"
                E(f"    mn{i}_{j} = {expr}")
        # trace(A @ mn)
        terms = [f"a{i}_{l} * mn{l}_{i}" for i in range(N) for l in range(N)]
        E(f"    tr = " + " + ".join(terms))
        E(f"    c{N - kk} = -tr * {1.0/kk}")
        # m = mn
        for i in range(N):
            for j in range(N):
                E(f"    m{i}_{j} = mn{i}_{j}")

    # Phase 2: Laguerre + deflation
    E(f"    # Phase 2: Laguerre")
    # Get sorted diagonal
    diag_vars = [f"a{i}_{i}" for i in range(N)]
    for i in range(N):
        E(f"    d{i} = {diag_vars[i]}")
    # Bubble sort
    for _ in range(N):
        for j in range(N-1):
            E(f"    _sw = d{j} > d{j+1}")
            E(f"    _a, _b = d{j}, d{j+1}")
            E(f"    d{j} = tl.where(_sw, _b, _a)")
            E(f"    d{j+1} = tl.where(_sw, _a, _b)")

    # Perturbation
    for i in range(N):
        p = -1e-4 + 2e-4 * i / max(N-1, 1)
        E(f"    d{i} = d{i} + {p}")

    # Working coefficients
    for k in range(N+1):
        E(f"    w{k} = c{k} + 0.0")

    # Sequential roots
    for ri in range(N):
        deg = N - ri
        E(f"    # Root {ri} (deg={deg})")
        E(f"    z = d{ri}")
        for _lag in range(5):
            E(f"    pv = w{deg}")
            E(f"    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)")
            E(f"    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)")
            for j in range(deg-1, -1, -1):
                E(f"    d2 = d2 * z + dp")
                E(f"    dp = dp * z + pv")
                E(f"    pv = pv * z + w{j}")
            E(f"    ok = tl.abs(pv) > 1e-30")
            E(f"    ps = tl.where(ok, pv, 1.0)")
            E(f"    G = tl.where(ok, dp / ps, 0.0)")
            E(f"    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)")
            E(f"    disc = tl.maximum({deg-1.0} * ({float(deg)} * H - G * G), 0.0)")
            E(f"    sq = tl.sqrt(disc)")
            E(f"    gp = G + sq")
            E(f"    gm = G - sq")
            E(f"    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)")
            E(f"    dok = tl.abs(den) > 1e-20")
            E(f"    ds = tl.where(dok, den, 1.0)")
            E(f"    z = z - tl.where(dok, {float(deg)} / ds, 0.0)")
        E(f"    r{ri} = z")
        # Synthetic division
        if deg > 1:
            E(f"    _b = w{deg}")
            for j in range(deg-1, 0, -1):
                E(f"    _bn = w{j} + z * _b")
                E(f"    w{j} = _b")
                E(f"    _b = _bn")
            E(f"    w0 = _b")

    # Newton polish
    E(f"    # Newton polish")
    for _pol in range(3):
        for ri in range(N):
            E(f"    pv = c{N} + 0.0")
            E(f"    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)")
            for j in range(N-1, -1, -1):
                E(f"    dp = dp * r{ri} + pv")
                E(f"    pv = pv * r{ri} + c{j}")
            E(f"    ok = tl.abs(dp) > 1e-30")
            E(f"    ds = tl.where(ok, dp, 1.0)")
            E(f"    r{ri} = r{ri} - tl.where(ok, pv / ds, 0.0)")

    # Phase 3: Eigenvectors via interleaved FL + Horner
    E(f"    # Phase 3: Eigenvectors")
    for ei in range(N):
        E(f"    # Eigenvector {ei}")
        E(f"    lam = r{ei}")
        # Reset M to zero
        for i in range(N):
            for j in range(N):
                E(f"    m{i}_{j} = tl.zeros((BLOCK_B,), dtype=tl.float64)")
        # Interleaved FL + Horner
        for kk in range(1, N+1):
            c_diag = f"c{N - kk + 1}"
            for i in range(N):
                for j in range(N):
                    terms = [f"a{i}_{l} * m{l}_{j}" for l in range(N)]
                    expr = " + ".join(terms)
                    if i == j:
                        expr += f" + {c_diag}"
                    E(f"    mn{i}_{j} = {expr}")
            if kk == 1:
                for i in range(N):
                    for j in range(N):
                        E(f"    h{i}_{j} = mn{i}_{j}")
            else:
                for i in range(N):
                    for j in range(N):
                        E(f"    h{i}_{j} = h{i}_{j} * lam + mn{i}_{j}")
            for i in range(N):
                for j in range(N):
                    E(f"    m{i}_{j} = mn{i}_{j}")

        # Max-norm column
        E(f"    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)")
        E(f"    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0")
        for j in range(N):
            terms = [f"h{i}_{j} * h{i}_{j}" for i in range(N)]
            E(f"    _cn = " + " + ".join(terms))
            E(f"    _better = _cn > best_n2")
            E(f"    best_n2 = tl.where(_better, _cn, best_n2)")
            E(f"    best_j = tl.where(_better, {j}, best_j)")
        # Extract
        for i in range(N):
            E(f"    ev{ei}_{i} = tl.zeros((BLOCK_B,), dtype=tl.float64)")
            for j in range(N):
                E(f"    ev{ei}_{i} = tl.where(best_j == {j}, h{i}_{j}, ev{ei}_{i})")
        # Normalize
        terms = [f"ev{ei}_{i} * ev{ei}_{i}" for i in range(N)]
        E(f"    _vn = tl.sqrt(" + " + ".join(terms) + " + 1e-60)")
        for i in range(N):
            E(f"    ev{ei}_{i} = ev{ei}_{i} / _vn")

    # Phase 4: NS (fp32) — V[i,j] = ev{j}_{i} (column j, row i)
    E(f"    # Phase 4: NS")
    for i in range(N):
        for j in range(N):
            E(f"    v{i}_{j} = ev{j}_{i}.to(tl.float32)")

    for _ns in range(2):
        E(f"    # NS iter")
        # Y = V^T V
        for i in range(N):
            for j in range(N):
                terms = [f"v{l}_{i} * v{l}_{j}" for l in range(N)]
                E(f"    y{i}_{j} = " + " + ".join(terms))
        # T = 3I - Y
        for i in range(N):
            for j in range(N):
                if i == j:
                    E(f"    t{i}_{j} = 3.0 - y{i}_{j}")
                else:
                    E(f"    t{i}_{j} = -y{i}_{j}")
        # V_new = 0.5 * V @ T
        for i in range(N):
            for j in range(N):
                terms = [f"v{i}_{l} * t{l}_{j}" for l in range(N)]
                E(f"    vn{i}_{j} = 0.5 * (" + " + ".join(terms) + ")")
        for i in range(N):
            for j in range(N):
                E(f"    v{i}_{j} = vn{i}_{j}")

    # Phase 5: Rayleigh (on pre-scaled A, then un-scale)
    E(f"    # Phase 5: Rayleigh")
    for i in range(N):
        for j in range(N):
            E(f"    af{i}_{j} = a{i}_{j}.to(tl.float32)")

    evals_vars = []
    for ei in range(N):
        E(f"    lam{ei} = tl.zeros((BLOCK_B,), dtype=tl.float32)")
        for l in range(N):
            terms = [f"af{l}_{mm} * v{mm}_{ei}" for mm in range(N)]
            E(f"    _av = " + " + ".join(terms))
            E(f"    lam{ei} = lam{ei} + v{l}_{ei} * _av")
        E(f"    lam{ei} = lam{ei} * sc.to(tl.float32)")
        evals_vars.append(f"lam{ei}")

    # Sort
    E(f"    # Sort")
    perm_vars = [f"p{i}" for i in range(N)]
    for i in range(N):
        E(f"    p{i} = tl.zeros((BLOCK_B,), dtype=tl.int32) + {i}")
    for _ in range(N):
        for j in range(N-1):
            E(f"    _sw = {evals_vars[j]} > {evals_vars[j+1]}")
            E(f"    _ea, _eb = {evals_vars[j]}, {evals_vars[j+1]}")
            E(f"    {evals_vars[j]} = tl.where(_sw, _eb, _ea)")
            E(f"    {evals_vars[j+1]} = tl.where(_sw, _ea, _eb)")
            E(f"    _pa, _pb = {perm_vars[j]}, {perm_vars[j+1]}")
            E(f"    {perm_vars[j]} = tl.where(_sw, _pb, _pa)")
            E(f"    {perm_vars[j+1]} = tl.where(_sw, _pa, _pb)")

    # Permute eigenvectors and store
    E(f"    # Store")
    for j_out in range(N):
        for j_src in range(N):
            E(f"    _is{j_src} = ({perm_vars[j_out]} == {j_src})")
        for i in range(N):
            E(f"    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)")
            for j_src in range(N):
                E(f"    _sv = tl.where(_is{j_src}, v{i}_{j_src}, _sv)")
            E(f"    tl.store(evecs_ptr + bid * {N2} + {i*N+j_out}, _sv, mask=mask)")

    for ei in range(N):
        E(f"    tl.store(evals_ptr + bid * {N} + {ei}, {evals_vars[ei]}, mask=mask)")

    return "\n".join(L)


# Generate kernel source and write to temp file (Triton needs inspect.getsource)
_kernel_src = _gen_kernel(6)
_kernel_path = '/tmp/_fl_eigh_triton_gen.py'
with open(_kernel_path, 'w') as _f:
    _f.write("import triton\nimport triton.language as tl\n\n")
    _f.write(_kernel_src)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_fl_eigh_triton_gen", _kernel_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_fl_eigh_gen = _mod._fl_eigh_gen


class FLEighTriton:
    @staticmethod
    def apply(A: Tensor) -> Tuple[Tensor, Tensor]:
        B, n, _ = A.shape
        evals = torch.empty(B, n, device=A.device, dtype=torch.float32)
        evecs = torch.empty(B, n, n, device=A.device, dtype=torch.float32)
        BLOCK_B = 32
        grid = ((B + BLOCK_B - 1) // BLOCK_B,)
        _fl_eigh_gen[grid](A.contiguous(), evals, evecs, B, BLOCK_B)
        return evals, evecs


# Benchmark
def sync(): torch.cuda.synchronize()
def gt(fn, w=20, r=200):
    for _ in range(w): fn()
    sync(); t = time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter() - t) / r
def fmt(s):
    if s < 1e-3: return f"{s*1e6:.1f}us"
    if s < 1: return f"{s*1e3:.2f}ms"
    return f"{s:.3f}s"


def main():
    if not torch.cuda.is_available(): sys.exit(1)
    dev = torch.device('cuda')
    p = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print("  FL Eigh Triton — Generated Kernel")
    print("=" * 72)
    print(f"  {p.name} | Triton {triton.__version__}")
    print(f"  Generated kernel: {len(_kernel_src.splitlines())} lines")

    N = 6; B = 4096
    A = (lambda R: (R + R.mT) / 2)(torch.randn(B, N, N, device=dev))
    rv, rV = torch.linalg.eigh(A)

    print(f"\n  ACCURACY (n={N} B={B})")
    try:
        tv, tV = FLEighTriton.apply(A)
        ve = (tv - rv).abs().max().item()
        dots = torch.bmm(rV.double().mT, tV.double()).abs().max(dim=-1).values.min().item()
        AV = torch.bmm(A.double(), tV.double())
        VL = tV.double() * tv.double().unsqueeze(-2)
        res = (AV - VL).reshape(B, -1).norm(dim=-1) / A.double().reshape(B, -1).norm(dim=-1).clamp(min=1e-30)
        print(f"  Triton: val={ve:.1e} align={dots:.6f} res={res.max().item():.1e}")
        triton_ok = True
    except Exception as e:
        print(f"  FAILED: {str(e)[:300]}")
        triton_ok = False

    print(f"\n  THROUGHPUT (n={N} B={B})")
    tr = gt(lambda: torch.linalg.eigh(A))
    print(f"  cuSOLVER: {fmt(tr)}")

    if triton_ok:
        tt = gt(lambda: FLEighTriton.apply(A))
        print(f"  Triton:   {fmt(tt)} ({tr/tt:.2f}x)")

        print(f"\n  BATCH SCALING (n={N})")
        print(f"  {'B':>6}  {'cuSOLVER':>10}  {'Triton':>10}  {'ratio':>7}")
        for Bx in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            Ax = (lambda R: (R+R.mT)/2)(torch.randn(Bx, N, N, device=dev))
            t1 = gt(lambda: torch.linalg.eigh(Ax), 10, 100)
            t2 = gt(lambda: FLEighTriton.apply(Ax), 10, 100)
            print(f"  {Bx:>6}  {fmt(t1):>10}  {fmt(t2):>10}  {t1/t2:>6.2f}x")
            del Ax

    print("=" * 72)


if __name__ == '__main__':
    main()