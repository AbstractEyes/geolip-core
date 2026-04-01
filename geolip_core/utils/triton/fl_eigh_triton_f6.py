"""Auto-generated FL Eigh Triton kernel for n=6. Do not edit."""
import triton
import triton.language as tl

@triton.jit
def _fl_eigh_gen(A_ptr, evals_ptr, evecs_ptr, B, BLOCK_B: tl.constexpr):
    pid = tl.program_id(0)
    bid = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = bid < B
    off = bid * 36
    # Load A
    frob = tl.zeros((BLOCK_B,), dtype=tl.float64)
    a0_0 = tl.load(A_ptr + off + 0, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a0_0 * a0_0
    a0_1 = tl.load(A_ptr + off + 1, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a0_1 * a0_1
    a0_2 = tl.load(A_ptr + off + 2, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a0_2 * a0_2
    a0_3 = tl.load(A_ptr + off + 3, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a0_3 * a0_3
    a0_4 = tl.load(A_ptr + off + 4, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a0_4 * a0_4
    a0_5 = tl.load(A_ptr + off + 5, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a0_5 * a0_5
    a1_0 = tl.load(A_ptr + off + 6, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a1_0 * a1_0
    a1_1 = tl.load(A_ptr + off + 7, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a1_1 * a1_1
    a1_2 = tl.load(A_ptr + off + 8, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a1_2 * a1_2
    a1_3 = tl.load(A_ptr + off + 9, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a1_3 * a1_3
    a1_4 = tl.load(A_ptr + off + 10, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a1_4 * a1_4
    a1_5 = tl.load(A_ptr + off + 11, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a1_5 * a1_5
    a2_0 = tl.load(A_ptr + off + 12, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a2_0 * a2_0
    a2_1 = tl.load(A_ptr + off + 13, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a2_1 * a2_1
    a2_2 = tl.load(A_ptr + off + 14, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a2_2 * a2_2
    a2_3 = tl.load(A_ptr + off + 15, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a2_3 * a2_3
    a2_4 = tl.load(A_ptr + off + 16, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a2_4 * a2_4
    a2_5 = tl.load(A_ptr + off + 17, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a2_5 * a2_5
    a3_0 = tl.load(A_ptr + off + 18, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a3_0 * a3_0
    a3_1 = tl.load(A_ptr + off + 19, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a3_1 * a3_1
    a3_2 = tl.load(A_ptr + off + 20, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a3_2 * a3_2
    a3_3 = tl.load(A_ptr + off + 21, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a3_3 * a3_3
    a3_4 = tl.load(A_ptr + off + 22, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a3_4 * a3_4
    a3_5 = tl.load(A_ptr + off + 23, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a3_5 * a3_5
    a4_0 = tl.load(A_ptr + off + 24, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a4_0 * a4_0
    a4_1 = tl.load(A_ptr + off + 25, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a4_1 * a4_1
    a4_2 = tl.load(A_ptr + off + 26, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a4_2 * a4_2
    a4_3 = tl.load(A_ptr + off + 27, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a4_3 * a4_3
    a4_4 = tl.load(A_ptr + off + 28, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a4_4 * a4_4
    a4_5 = tl.load(A_ptr + off + 29, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a4_5 * a4_5
    a5_0 = tl.load(A_ptr + off + 30, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a5_0 * a5_0
    a5_1 = tl.load(A_ptr + off + 31, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a5_1 * a5_1
    a5_2 = tl.load(A_ptr + off + 32, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a5_2 * a5_2
    a5_3 = tl.load(A_ptr + off + 33, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a5_3 * a5_3
    a5_4 = tl.load(A_ptr + off + 34, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a5_4 * a5_4
    a5_5 = tl.load(A_ptr + off + 35, mask=mask, other=0.0).to(tl.float64)
    frob = frob + a5_5 * a5_5
    sc = tl.sqrt(frob * 0.16666666666666666)
    sc = tl.where(sc > 1e-12, sc, 1e-12)
    inv_sc = 1.0 / sc
    a0_0 = a0_0 * inv_sc
    a0_1 = a0_1 * inv_sc
    a0_2 = a0_2 * inv_sc
    a0_3 = a0_3 * inv_sc
    a0_4 = a0_4 * inv_sc
    a0_5 = a0_5 * inv_sc
    a1_0 = a1_0 * inv_sc
    a1_1 = a1_1 * inv_sc
    a1_2 = a1_2 * inv_sc
    a1_3 = a1_3 * inv_sc
    a1_4 = a1_4 * inv_sc
    a1_5 = a1_5 * inv_sc
    a2_0 = a2_0 * inv_sc
    a2_1 = a2_1 * inv_sc
    a2_2 = a2_2 * inv_sc
    a2_3 = a2_3 * inv_sc
    a2_4 = a2_4 * inv_sc
    a2_5 = a2_5 * inv_sc
    a3_0 = a3_0 * inv_sc
    a3_1 = a3_1 * inv_sc
    a3_2 = a3_2 * inv_sc
    a3_3 = a3_3 * inv_sc
    a3_4 = a3_4 * inv_sc
    a3_5 = a3_5 * inv_sc
    a4_0 = a4_0 * inv_sc
    a4_1 = a4_1 * inv_sc
    a4_2 = a4_2 * inv_sc
    a4_3 = a4_3 * inv_sc
    a4_4 = a4_4 * inv_sc
    a4_5 = a4_5 * inv_sc
    a5_0 = a5_0 * inv_sc
    a5_1 = a5_1 * inv_sc
    a5_2 = a5_2 * inv_sc
    a5_3 = a5_3 * inv_sc
    a5_4 = a5_4 * inv_sc
    a5_5 = a5_5 * inv_sc
    # Phase 1: FL coefficients
    c0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    c1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    c2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    c3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    c4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    c5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    c6 = tl.zeros((BLOCK_B,), dtype=tl.float64) + 1.0
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    # FL iter k=1
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    tr = a0_0 * mn0_0 + a0_1 * mn1_0 + a0_2 * mn2_0 + a0_3 * mn3_0 + a0_4 * mn4_0 + a0_5 * mn5_0 + a1_0 * mn0_1 + a1_1 * mn1_1 + a1_2 * mn2_1 + a1_3 * mn3_1 + a1_4 * mn4_1 + a1_5 * mn5_1 + a2_0 * mn0_2 + a2_1 * mn1_2 + a2_2 * mn2_2 + a2_3 * mn3_2 + a2_4 * mn4_2 + a2_5 * mn5_2 + a3_0 * mn0_3 + a3_1 * mn1_3 + a3_2 * mn2_3 + a3_3 * mn3_3 + a3_4 * mn4_3 + a3_5 * mn5_3 + a4_0 * mn0_4 + a4_1 * mn1_4 + a4_2 * mn2_4 + a4_3 * mn3_4 + a4_4 * mn4_4 + a4_5 * mn5_4 + a5_0 * mn0_5 + a5_1 * mn1_5 + a5_2 * mn2_5 + a5_3 * mn3_5 + a5_4 * mn4_5 + a5_5 * mn5_5
    c5 = -tr * 1.0
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    # FL iter k=2
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    tr = a0_0 * mn0_0 + a0_1 * mn1_0 + a0_2 * mn2_0 + a0_3 * mn3_0 + a0_4 * mn4_0 + a0_5 * mn5_0 + a1_0 * mn0_1 + a1_1 * mn1_1 + a1_2 * mn2_1 + a1_3 * mn3_1 + a1_4 * mn4_1 + a1_5 * mn5_1 + a2_0 * mn0_2 + a2_1 * mn1_2 + a2_2 * mn2_2 + a2_3 * mn3_2 + a2_4 * mn4_2 + a2_5 * mn5_2 + a3_0 * mn0_3 + a3_1 * mn1_3 + a3_2 * mn2_3 + a3_3 * mn3_3 + a3_4 * mn4_3 + a3_5 * mn5_3 + a4_0 * mn0_4 + a4_1 * mn1_4 + a4_2 * mn2_4 + a4_3 * mn3_4 + a4_4 * mn4_4 + a4_5 * mn5_4 + a5_0 * mn0_5 + a5_1 * mn1_5 + a5_2 * mn2_5 + a5_3 * mn3_5 + a5_4 * mn4_5 + a5_5 * mn5_5
    c4 = -tr * 0.5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    # FL iter k=3
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    tr = a0_0 * mn0_0 + a0_1 * mn1_0 + a0_2 * mn2_0 + a0_3 * mn3_0 + a0_4 * mn4_0 + a0_5 * mn5_0 + a1_0 * mn0_1 + a1_1 * mn1_1 + a1_2 * mn2_1 + a1_3 * mn3_1 + a1_4 * mn4_1 + a1_5 * mn5_1 + a2_0 * mn0_2 + a2_1 * mn1_2 + a2_2 * mn2_2 + a2_3 * mn3_2 + a2_4 * mn4_2 + a2_5 * mn5_2 + a3_0 * mn0_3 + a3_1 * mn1_3 + a3_2 * mn2_3 + a3_3 * mn3_3 + a3_4 * mn4_3 + a3_5 * mn5_3 + a4_0 * mn0_4 + a4_1 * mn1_4 + a4_2 * mn2_4 + a4_3 * mn3_4 + a4_4 * mn4_4 + a4_5 * mn5_4 + a5_0 * mn0_5 + a5_1 * mn1_5 + a5_2 * mn2_5 + a5_3 * mn3_5 + a5_4 * mn4_5 + a5_5 * mn5_5
    c3 = -tr * 0.3333333333333333
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    # FL iter k=4
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    tr = a0_0 * mn0_0 + a0_1 * mn1_0 + a0_2 * mn2_0 + a0_3 * mn3_0 + a0_4 * mn4_0 + a0_5 * mn5_0 + a1_0 * mn0_1 + a1_1 * mn1_1 + a1_2 * mn2_1 + a1_3 * mn3_1 + a1_4 * mn4_1 + a1_5 * mn5_1 + a2_0 * mn0_2 + a2_1 * mn1_2 + a2_2 * mn2_2 + a2_3 * mn3_2 + a2_4 * mn4_2 + a2_5 * mn5_2 + a3_0 * mn0_3 + a3_1 * mn1_3 + a3_2 * mn2_3 + a3_3 * mn3_3 + a3_4 * mn4_3 + a3_5 * mn5_3 + a4_0 * mn0_4 + a4_1 * mn1_4 + a4_2 * mn2_4 + a4_3 * mn3_4 + a4_4 * mn4_4 + a4_5 * mn5_4 + a5_0 * mn0_5 + a5_1 * mn1_5 + a5_2 * mn2_5 + a5_3 * mn3_5 + a5_4 * mn4_5 + a5_5 * mn5_5
    c2 = -tr * 0.25
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    # FL iter k=5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    tr = a0_0 * mn0_0 + a0_1 * mn1_0 + a0_2 * mn2_0 + a0_3 * mn3_0 + a0_4 * mn4_0 + a0_5 * mn5_0 + a1_0 * mn0_1 + a1_1 * mn1_1 + a1_2 * mn2_1 + a1_3 * mn3_1 + a1_4 * mn4_1 + a1_5 * mn5_1 + a2_0 * mn0_2 + a2_1 * mn1_2 + a2_2 * mn2_2 + a2_3 * mn3_2 + a2_4 * mn4_2 + a2_5 * mn5_2 + a3_0 * mn0_3 + a3_1 * mn1_3 + a3_2 * mn2_3 + a3_3 * mn3_3 + a3_4 * mn4_3 + a3_5 * mn5_3 + a4_0 * mn0_4 + a4_1 * mn1_4 + a4_2 * mn2_4 + a4_3 * mn3_4 + a4_4 * mn4_4 + a4_5 * mn5_4 + a5_0 * mn0_5 + a5_1 * mn1_5 + a5_2 * mn2_5 + a5_3 * mn3_5 + a5_4 * mn4_5 + a5_5 * mn5_5
    c1 = -tr * 0.2
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    # FL iter k=6
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    tr = a0_0 * mn0_0 + a0_1 * mn1_0 + a0_2 * mn2_0 + a0_3 * mn3_0 + a0_4 * mn4_0 + a0_5 * mn5_0 + a1_0 * mn0_1 + a1_1 * mn1_1 + a1_2 * mn2_1 + a1_3 * mn3_1 + a1_4 * mn4_1 + a1_5 * mn5_1 + a2_0 * mn0_2 + a2_1 * mn1_2 + a2_2 * mn2_2 + a2_3 * mn3_2 + a2_4 * mn4_2 + a2_5 * mn5_2 + a3_0 * mn0_3 + a3_1 * mn1_3 + a3_2 * mn2_3 + a3_3 * mn3_3 + a3_4 * mn4_3 + a3_5 * mn5_3 + a4_0 * mn0_4 + a4_1 * mn1_4 + a4_2 * mn2_4 + a4_3 * mn3_4 + a4_4 * mn4_4 + a4_5 * mn5_4 + a5_0 * mn0_5 + a5_1 * mn1_5 + a5_2 * mn2_5 + a5_3 * mn3_5 + a5_4 * mn4_5 + a5_5 * mn5_5
    c0 = -tr * 0.16666666666666666
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    # Phase 2: Laguerre
    d0 = a0_0
    d1 = a1_1
    d2 = a2_2
    d3 = a3_3
    d4 = a4_4
    d5 = a5_5
    _sw = d0 > d1
    _a, _b = d0, d1
    d0 = tl.where(_sw, _b, _a)
    d1 = tl.where(_sw, _a, _b)
    _sw = d1 > d2
    _a, _b = d1, d2
    d1 = tl.where(_sw, _b, _a)
    d2 = tl.where(_sw, _a, _b)
    _sw = d2 > d3
    _a, _b = d2, d3
    d2 = tl.where(_sw, _b, _a)
    d3 = tl.where(_sw, _a, _b)
    _sw = d3 > d4
    _a, _b = d3, d4
    d3 = tl.where(_sw, _b, _a)
    d4 = tl.where(_sw, _a, _b)
    _sw = d4 > d5
    _a, _b = d4, d5
    d4 = tl.where(_sw, _b, _a)
    d5 = tl.where(_sw, _a, _b)
    _sw = d0 > d1
    _a, _b = d0, d1
    d0 = tl.where(_sw, _b, _a)
    d1 = tl.where(_sw, _a, _b)
    _sw = d1 > d2
    _a, _b = d1, d2
    d1 = tl.where(_sw, _b, _a)
    d2 = tl.where(_sw, _a, _b)
    _sw = d2 > d3
    _a, _b = d2, d3
    d2 = tl.where(_sw, _b, _a)
    d3 = tl.where(_sw, _a, _b)
    _sw = d3 > d4
    _a, _b = d3, d4
    d3 = tl.where(_sw, _b, _a)
    d4 = tl.where(_sw, _a, _b)
    _sw = d4 > d5
    _a, _b = d4, d5
    d4 = tl.where(_sw, _b, _a)
    d5 = tl.where(_sw, _a, _b)
    _sw = d0 > d1
    _a, _b = d0, d1
    d0 = tl.where(_sw, _b, _a)
    d1 = tl.where(_sw, _a, _b)
    _sw = d1 > d2
    _a, _b = d1, d2
    d1 = tl.where(_sw, _b, _a)
    d2 = tl.where(_sw, _a, _b)
    _sw = d2 > d3
    _a, _b = d2, d3
    d2 = tl.where(_sw, _b, _a)
    d3 = tl.where(_sw, _a, _b)
    _sw = d3 > d4
    _a, _b = d3, d4
    d3 = tl.where(_sw, _b, _a)
    d4 = tl.where(_sw, _a, _b)
    _sw = d4 > d5
    _a, _b = d4, d5
    d4 = tl.where(_sw, _b, _a)
    d5 = tl.where(_sw, _a, _b)
    _sw = d0 > d1
    _a, _b = d0, d1
    d0 = tl.where(_sw, _b, _a)
    d1 = tl.where(_sw, _a, _b)
    _sw = d1 > d2
    _a, _b = d1, d2
    d1 = tl.where(_sw, _b, _a)
    d2 = tl.where(_sw, _a, _b)
    _sw = d2 > d3
    _a, _b = d2, d3
    d2 = tl.where(_sw, _b, _a)
    d3 = tl.where(_sw, _a, _b)
    _sw = d3 > d4
    _a, _b = d3, d4
    d3 = tl.where(_sw, _b, _a)
    d4 = tl.where(_sw, _a, _b)
    _sw = d4 > d5
    _a, _b = d4, d5
    d4 = tl.where(_sw, _b, _a)
    d5 = tl.where(_sw, _a, _b)
    _sw = d0 > d1
    _a, _b = d0, d1
    d0 = tl.where(_sw, _b, _a)
    d1 = tl.where(_sw, _a, _b)
    _sw = d1 > d2
    _a, _b = d1, d2
    d1 = tl.where(_sw, _b, _a)
    d2 = tl.where(_sw, _a, _b)
    _sw = d2 > d3
    _a, _b = d2, d3
    d2 = tl.where(_sw, _b, _a)
    d3 = tl.where(_sw, _a, _b)
    _sw = d3 > d4
    _a, _b = d3, d4
    d3 = tl.where(_sw, _b, _a)
    d4 = tl.where(_sw, _a, _b)
    _sw = d4 > d5
    _a, _b = d4, d5
    d4 = tl.where(_sw, _b, _a)
    d5 = tl.where(_sw, _a, _b)
    _sw = d0 > d1
    _a, _b = d0, d1
    d0 = tl.where(_sw, _b, _a)
    d1 = tl.where(_sw, _a, _b)
    _sw = d1 > d2
    _a, _b = d1, d2
    d1 = tl.where(_sw, _b, _a)
    d2 = tl.where(_sw, _a, _b)
    _sw = d2 > d3
    _a, _b = d2, d3
    d2 = tl.where(_sw, _b, _a)
    d3 = tl.where(_sw, _a, _b)
    _sw = d3 > d4
    _a, _b = d3, d4
    d3 = tl.where(_sw, _b, _a)
    d4 = tl.where(_sw, _a, _b)
    _sw = d4 > d5
    _a, _b = d4, d5
    d4 = tl.where(_sw, _b, _a)
    d5 = tl.where(_sw, _a, _b)
    d0 = d0 + -0.0001
    d1 = d1 + -6e-05
    d2 = d2 + -1.9999999999999998e-05
    d3 = d3 + 2.0000000000000012e-05
    d4 = d4 + 6.000000000000001e-05
    d5 = d5 + 0.0001
    w0 = c0 + 0.0
    w1 = c1 + 0.0
    w2 = c2 + 0.0
    w3 = c3 + 0.0
    w4 = c4 + 0.0
    w5 = c5 + 0.0
    w6 = c6 + 0.0
    # Root 0 (deg=6)
    z = d0
    pv = w6
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w5
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(5.0 * (6.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 6.0 / ds, 0.0)
    pv = w6
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w5
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(5.0 * (6.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 6.0 / ds, 0.0)
    pv = w6
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w5
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(5.0 * (6.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 6.0 / ds, 0.0)
    pv = w6
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w5
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(5.0 * (6.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 6.0 / ds, 0.0)
    pv = w6
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w5
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(5.0 * (6.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 6.0 / ds, 0.0)
    r0 = z
    _b = w6
    _bn = w5 + z * _b
    w5 = _b
    _b = _bn
    _bn = w4 + z * _b
    w4 = _b
    _b = _bn
    _bn = w3 + z * _b
    w3 = _b
    _b = _bn
    _bn = w2 + z * _b
    w2 = _b
    _b = _bn
    _bn = w1 + z * _b
    w1 = _b
    _b = _bn
    w0 = _b
    # Root 1 (deg=5)
    z = d1
    pv = w5
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(4.0 * (5.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 5.0 / ds, 0.0)
    pv = w5
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(4.0 * (5.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 5.0 / ds, 0.0)
    pv = w5
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(4.0 * (5.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 5.0 / ds, 0.0)
    pv = w5
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(4.0 * (5.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 5.0 / ds, 0.0)
    pv = w5
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w4
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(4.0 * (5.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 5.0 / ds, 0.0)
    r1 = z
    _b = w5
    _bn = w4 + z * _b
    w4 = _b
    _b = _bn
    _bn = w3 + z * _b
    w3 = _b
    _b = _bn
    _bn = w2 + z * _b
    w2 = _b
    _b = _bn
    _bn = w1 + z * _b
    w1 = _b
    _b = _bn
    w0 = _b
    # Root 2 (deg=4)
    z = d2
    pv = w4
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(3.0 * (4.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 4.0 / ds, 0.0)
    pv = w4
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(3.0 * (4.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 4.0 / ds, 0.0)
    pv = w4
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(3.0 * (4.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 4.0 / ds, 0.0)
    pv = w4
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(3.0 * (4.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 4.0 / ds, 0.0)
    pv = w4
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w3
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(3.0 * (4.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 4.0 / ds, 0.0)
    r2 = z
    _b = w4
    _bn = w3 + z * _b
    w3 = _b
    _b = _bn
    _bn = w2 + z * _b
    w2 = _b
    _b = _bn
    _bn = w1 + z * _b
    w1 = _b
    _b = _bn
    w0 = _b
    # Root 3 (deg=3)
    z = d3
    pv = w3
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(2.0 * (3.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 3.0 / ds, 0.0)
    pv = w3
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(2.0 * (3.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 3.0 / ds, 0.0)
    pv = w3
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(2.0 * (3.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 3.0 / ds, 0.0)
    pv = w3
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(2.0 * (3.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 3.0 / ds, 0.0)
    pv = w3
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w2
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(2.0 * (3.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 3.0 / ds, 0.0)
    r3 = z
    _b = w3
    _bn = w2 + z * _b
    w2 = _b
    _b = _bn
    _bn = w1 + z * _b
    w1 = _b
    _b = _bn
    w0 = _b
    # Root 4 (deg=2)
    z = d4
    pv = w2
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(1.0 * (2.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 2.0 / ds, 0.0)
    pv = w2
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(1.0 * (2.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 2.0 / ds, 0.0)
    pv = w2
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(1.0 * (2.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 2.0 / ds, 0.0)
    pv = w2
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(1.0 * (2.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 2.0 / ds, 0.0)
    pv = w2
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w1
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(1.0 * (2.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 2.0 / ds, 0.0)
    r4 = z
    _b = w2
    _bn = w1 + z * _b
    w1 = _b
    _b = _bn
    w0 = _b
    # Root 5 (deg=1)
    z = d5
    pv = w1
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(0.0 * (1.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 1.0 / ds, 0.0)
    pv = w1
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(0.0 * (1.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 1.0 / ds, 0.0)
    pv = w1
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(0.0 * (1.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 1.0 / ds, 0.0)
    pv = w1
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(0.0 * (1.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 1.0 / ds, 0.0)
    pv = w1
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    d2 = d2 * z + dp
    dp = dp * z + pv
    pv = pv * z + w0
    ok = tl.abs(pv) > 1e-30
    ps = tl.where(ok, pv, 1.0)
    G = tl.where(ok, dp / ps, 0.0)
    H = G * G - tl.where(ok, 2.0 * d2 / ps, 0.0)
    disc = tl.maximum(0.0 * (1.0 * H - G * G), 0.0)
    sq = tl.sqrt(disc)
    gp = G + sq
    gm = G - sq
    den = tl.where(tl.abs(gp) >= tl.abs(gm), gp, gm)
    dok = tl.abs(den) > 1e-20
    ds = tl.where(dok, den, 1.0)
    z = z - tl.where(dok, 1.0 / ds, 0.0)
    r5 = z
    # Newton polish
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r0 + pv
    pv = pv * r0 + c5
    dp = dp * r0 + pv
    pv = pv * r0 + c4
    dp = dp * r0 + pv
    pv = pv * r0 + c3
    dp = dp * r0 + pv
    pv = pv * r0 + c2
    dp = dp * r0 + pv
    pv = pv * r0 + c1
    dp = dp * r0 + pv
    pv = pv * r0 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r0 = r0 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r1 + pv
    pv = pv * r1 + c5
    dp = dp * r1 + pv
    pv = pv * r1 + c4
    dp = dp * r1 + pv
    pv = pv * r1 + c3
    dp = dp * r1 + pv
    pv = pv * r1 + c2
    dp = dp * r1 + pv
    pv = pv * r1 + c1
    dp = dp * r1 + pv
    pv = pv * r1 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r1 = r1 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r2 + pv
    pv = pv * r2 + c5
    dp = dp * r2 + pv
    pv = pv * r2 + c4
    dp = dp * r2 + pv
    pv = pv * r2 + c3
    dp = dp * r2 + pv
    pv = pv * r2 + c2
    dp = dp * r2 + pv
    pv = pv * r2 + c1
    dp = dp * r2 + pv
    pv = pv * r2 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r2 = r2 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r3 + pv
    pv = pv * r3 + c5
    dp = dp * r3 + pv
    pv = pv * r3 + c4
    dp = dp * r3 + pv
    pv = pv * r3 + c3
    dp = dp * r3 + pv
    pv = pv * r3 + c2
    dp = dp * r3 + pv
    pv = pv * r3 + c1
    dp = dp * r3 + pv
    pv = pv * r3 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r3 = r3 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r4 + pv
    pv = pv * r4 + c5
    dp = dp * r4 + pv
    pv = pv * r4 + c4
    dp = dp * r4 + pv
    pv = pv * r4 + c3
    dp = dp * r4 + pv
    pv = pv * r4 + c2
    dp = dp * r4 + pv
    pv = pv * r4 + c1
    dp = dp * r4 + pv
    pv = pv * r4 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r4 = r4 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r5 + pv
    pv = pv * r5 + c5
    dp = dp * r5 + pv
    pv = pv * r5 + c4
    dp = dp * r5 + pv
    pv = pv * r5 + c3
    dp = dp * r5 + pv
    pv = pv * r5 + c2
    dp = dp * r5 + pv
    pv = pv * r5 + c1
    dp = dp * r5 + pv
    pv = pv * r5 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r5 = r5 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r0 + pv
    pv = pv * r0 + c5
    dp = dp * r0 + pv
    pv = pv * r0 + c4
    dp = dp * r0 + pv
    pv = pv * r0 + c3
    dp = dp * r0 + pv
    pv = pv * r0 + c2
    dp = dp * r0 + pv
    pv = pv * r0 + c1
    dp = dp * r0 + pv
    pv = pv * r0 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r0 = r0 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r1 + pv
    pv = pv * r1 + c5
    dp = dp * r1 + pv
    pv = pv * r1 + c4
    dp = dp * r1 + pv
    pv = pv * r1 + c3
    dp = dp * r1 + pv
    pv = pv * r1 + c2
    dp = dp * r1 + pv
    pv = pv * r1 + c1
    dp = dp * r1 + pv
    pv = pv * r1 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r1 = r1 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r2 + pv
    pv = pv * r2 + c5
    dp = dp * r2 + pv
    pv = pv * r2 + c4
    dp = dp * r2 + pv
    pv = pv * r2 + c3
    dp = dp * r2 + pv
    pv = pv * r2 + c2
    dp = dp * r2 + pv
    pv = pv * r2 + c1
    dp = dp * r2 + pv
    pv = pv * r2 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r2 = r2 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r3 + pv
    pv = pv * r3 + c5
    dp = dp * r3 + pv
    pv = pv * r3 + c4
    dp = dp * r3 + pv
    pv = pv * r3 + c3
    dp = dp * r3 + pv
    pv = pv * r3 + c2
    dp = dp * r3 + pv
    pv = pv * r3 + c1
    dp = dp * r3 + pv
    pv = pv * r3 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r3 = r3 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r4 + pv
    pv = pv * r4 + c5
    dp = dp * r4 + pv
    pv = pv * r4 + c4
    dp = dp * r4 + pv
    pv = pv * r4 + c3
    dp = dp * r4 + pv
    pv = pv * r4 + c2
    dp = dp * r4 + pv
    pv = pv * r4 + c1
    dp = dp * r4 + pv
    pv = pv * r4 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r4 = r4 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r5 + pv
    pv = pv * r5 + c5
    dp = dp * r5 + pv
    pv = pv * r5 + c4
    dp = dp * r5 + pv
    pv = pv * r5 + c3
    dp = dp * r5 + pv
    pv = pv * r5 + c2
    dp = dp * r5 + pv
    pv = pv * r5 + c1
    dp = dp * r5 + pv
    pv = pv * r5 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r5 = r5 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r0 + pv
    pv = pv * r0 + c5
    dp = dp * r0 + pv
    pv = pv * r0 + c4
    dp = dp * r0 + pv
    pv = pv * r0 + c3
    dp = dp * r0 + pv
    pv = pv * r0 + c2
    dp = dp * r0 + pv
    pv = pv * r0 + c1
    dp = dp * r0 + pv
    pv = pv * r0 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r0 = r0 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r1 + pv
    pv = pv * r1 + c5
    dp = dp * r1 + pv
    pv = pv * r1 + c4
    dp = dp * r1 + pv
    pv = pv * r1 + c3
    dp = dp * r1 + pv
    pv = pv * r1 + c2
    dp = dp * r1 + pv
    pv = pv * r1 + c1
    dp = dp * r1 + pv
    pv = pv * r1 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r1 = r1 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r2 + pv
    pv = pv * r2 + c5
    dp = dp * r2 + pv
    pv = pv * r2 + c4
    dp = dp * r2 + pv
    pv = pv * r2 + c3
    dp = dp * r2 + pv
    pv = pv * r2 + c2
    dp = dp * r2 + pv
    pv = pv * r2 + c1
    dp = dp * r2 + pv
    pv = pv * r2 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r2 = r2 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r3 + pv
    pv = pv * r3 + c5
    dp = dp * r3 + pv
    pv = pv * r3 + c4
    dp = dp * r3 + pv
    pv = pv * r3 + c3
    dp = dp * r3 + pv
    pv = pv * r3 + c2
    dp = dp * r3 + pv
    pv = pv * r3 + c1
    dp = dp * r3 + pv
    pv = pv * r3 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r3 = r3 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r4 + pv
    pv = pv * r4 + c5
    dp = dp * r4 + pv
    pv = pv * r4 + c4
    dp = dp * r4 + pv
    pv = pv * r4 + c3
    dp = dp * r4 + pv
    pv = pv * r4 + c2
    dp = dp * r4 + pv
    pv = pv * r4 + c1
    dp = dp * r4 + pv
    pv = pv * r4 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r4 = r4 - tl.where(ok, pv / ds, 0.0)
    pv = c6 + 0.0
    dp = tl.zeros((BLOCK_B,), dtype=tl.float64)
    dp = dp * r5 + pv
    pv = pv * r5 + c5
    dp = dp * r5 + pv
    pv = pv * r5 + c4
    dp = dp * r5 + pv
    pv = pv * r5 + c3
    dp = dp * r5 + pv
    pv = pv * r5 + c2
    dp = dp * r5 + pv
    pv = pv * r5 + c1
    dp = dp * r5 + pv
    pv = pv * r5 + c0
    ok = tl.abs(dp) > 1e-30
    ds = tl.where(ok, dp, 1.0)
    r5 = r5 - tl.where(ok, pv / ds, 0.0)
    # Phase 3: Eigenvectors
    # Eigenvector 0
    lam = r0
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    h0_0 = mn0_0
    h0_1 = mn0_1
    h0_2 = mn0_2
    h0_3 = mn0_3
    h0_4 = mn0_4
    h0_5 = mn0_5
    h1_0 = mn1_0
    h1_1 = mn1_1
    h1_2 = mn1_2
    h1_3 = mn1_3
    h1_4 = mn1_4
    h1_5 = mn1_5
    h2_0 = mn2_0
    h2_1 = mn2_1
    h2_2 = mn2_2
    h2_3 = mn2_3
    h2_4 = mn2_4
    h2_5 = mn2_5
    h3_0 = mn3_0
    h3_1 = mn3_1
    h3_2 = mn3_2
    h3_3 = mn3_3
    h3_4 = mn3_4
    h3_5 = mn3_5
    h4_0 = mn4_0
    h4_1 = mn4_1
    h4_2 = mn4_2
    h4_3 = mn4_3
    h4_4 = mn4_4
    h4_5 = mn4_5
    h5_0 = mn5_0
    h5_1 = mn5_1
    h5_2 = mn5_2
    h5_3 = mn5_3
    h5_4 = mn5_4
    h5_5 = mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)
    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0
    _cn = h0_0 * h0_0 + h1_0 * h1_0 + h2_0 * h2_0 + h3_0 * h3_0 + h4_0 * h4_0 + h5_0 * h5_0
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 0, best_j)
    _cn = h0_1 * h0_1 + h1_1 * h1_1 + h2_1 * h2_1 + h3_1 * h3_1 + h4_1 * h4_1 + h5_1 * h5_1
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 1, best_j)
    _cn = h0_2 * h0_2 + h1_2 * h1_2 + h2_2 * h2_2 + h3_2 * h3_2 + h4_2 * h4_2 + h5_2 * h5_2
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 2, best_j)
    _cn = h0_3 * h0_3 + h1_3 * h1_3 + h2_3 * h2_3 + h3_3 * h3_3 + h4_3 * h4_3 + h5_3 * h5_3
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 3, best_j)
    _cn = h0_4 * h0_4 + h1_4 * h1_4 + h2_4 * h2_4 + h3_4 * h3_4 + h4_4 * h4_4 + h5_4 * h5_4
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 4, best_j)
    _cn = h0_5 * h0_5 + h1_5 * h1_5 + h2_5 * h2_5 + h3_5 * h3_5 + h4_5 * h4_5 + h5_5 * h5_5
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 5, best_j)
    ev0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev0_0 = tl.where(best_j == 0, h0_0, ev0_0)
    ev0_0 = tl.where(best_j == 1, h0_1, ev0_0)
    ev0_0 = tl.where(best_j == 2, h0_2, ev0_0)
    ev0_0 = tl.where(best_j == 3, h0_3, ev0_0)
    ev0_0 = tl.where(best_j == 4, h0_4, ev0_0)
    ev0_0 = tl.where(best_j == 5, h0_5, ev0_0)
    ev0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev0_1 = tl.where(best_j == 0, h1_0, ev0_1)
    ev0_1 = tl.where(best_j == 1, h1_1, ev0_1)
    ev0_1 = tl.where(best_j == 2, h1_2, ev0_1)
    ev0_1 = tl.where(best_j == 3, h1_3, ev0_1)
    ev0_1 = tl.where(best_j == 4, h1_4, ev0_1)
    ev0_1 = tl.where(best_j == 5, h1_5, ev0_1)
    ev0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev0_2 = tl.where(best_j == 0, h2_0, ev0_2)
    ev0_2 = tl.where(best_j == 1, h2_1, ev0_2)
    ev0_2 = tl.where(best_j == 2, h2_2, ev0_2)
    ev0_2 = tl.where(best_j == 3, h2_3, ev0_2)
    ev0_2 = tl.where(best_j == 4, h2_4, ev0_2)
    ev0_2 = tl.where(best_j == 5, h2_5, ev0_2)
    ev0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev0_3 = tl.where(best_j == 0, h3_0, ev0_3)
    ev0_3 = tl.where(best_j == 1, h3_1, ev0_3)
    ev0_3 = tl.where(best_j == 2, h3_2, ev0_3)
    ev0_3 = tl.where(best_j == 3, h3_3, ev0_3)
    ev0_3 = tl.where(best_j == 4, h3_4, ev0_3)
    ev0_3 = tl.where(best_j == 5, h3_5, ev0_3)
    ev0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev0_4 = tl.where(best_j == 0, h4_0, ev0_4)
    ev0_4 = tl.where(best_j == 1, h4_1, ev0_4)
    ev0_4 = tl.where(best_j == 2, h4_2, ev0_4)
    ev0_4 = tl.where(best_j == 3, h4_3, ev0_4)
    ev0_4 = tl.where(best_j == 4, h4_4, ev0_4)
    ev0_4 = tl.where(best_j == 5, h4_5, ev0_4)
    ev0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev0_5 = tl.where(best_j == 0, h5_0, ev0_5)
    ev0_5 = tl.where(best_j == 1, h5_1, ev0_5)
    ev0_5 = tl.where(best_j == 2, h5_2, ev0_5)
    ev0_5 = tl.where(best_j == 3, h5_3, ev0_5)
    ev0_5 = tl.where(best_j == 4, h5_4, ev0_5)
    ev0_5 = tl.where(best_j == 5, h5_5, ev0_5)
    _vn = tl.sqrt(ev0_0 * ev0_0 + ev0_1 * ev0_1 + ev0_2 * ev0_2 + ev0_3 * ev0_3 + ev0_4 * ev0_4 + ev0_5 * ev0_5 + 1e-60)
    ev0_0 = ev0_0 / _vn
    ev0_1 = ev0_1 / _vn
    ev0_2 = ev0_2 / _vn
    ev0_3 = ev0_3 / _vn
    ev0_4 = ev0_4 / _vn
    ev0_5 = ev0_5 / _vn
    # Eigenvector 1
    lam = r1
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    h0_0 = mn0_0
    h0_1 = mn0_1
    h0_2 = mn0_2
    h0_3 = mn0_3
    h0_4 = mn0_4
    h0_5 = mn0_5
    h1_0 = mn1_0
    h1_1 = mn1_1
    h1_2 = mn1_2
    h1_3 = mn1_3
    h1_4 = mn1_4
    h1_5 = mn1_5
    h2_0 = mn2_0
    h2_1 = mn2_1
    h2_2 = mn2_2
    h2_3 = mn2_3
    h2_4 = mn2_4
    h2_5 = mn2_5
    h3_0 = mn3_0
    h3_1 = mn3_1
    h3_2 = mn3_2
    h3_3 = mn3_3
    h3_4 = mn3_4
    h3_5 = mn3_5
    h4_0 = mn4_0
    h4_1 = mn4_1
    h4_2 = mn4_2
    h4_3 = mn4_3
    h4_4 = mn4_4
    h4_5 = mn4_5
    h5_0 = mn5_0
    h5_1 = mn5_1
    h5_2 = mn5_2
    h5_3 = mn5_3
    h5_4 = mn5_4
    h5_5 = mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)
    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0
    _cn = h0_0 * h0_0 + h1_0 * h1_0 + h2_0 * h2_0 + h3_0 * h3_0 + h4_0 * h4_0 + h5_0 * h5_0
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 0, best_j)
    _cn = h0_1 * h0_1 + h1_1 * h1_1 + h2_1 * h2_1 + h3_1 * h3_1 + h4_1 * h4_1 + h5_1 * h5_1
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 1, best_j)
    _cn = h0_2 * h0_2 + h1_2 * h1_2 + h2_2 * h2_2 + h3_2 * h3_2 + h4_2 * h4_2 + h5_2 * h5_2
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 2, best_j)
    _cn = h0_3 * h0_3 + h1_3 * h1_3 + h2_3 * h2_3 + h3_3 * h3_3 + h4_3 * h4_3 + h5_3 * h5_3
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 3, best_j)
    _cn = h0_4 * h0_4 + h1_4 * h1_4 + h2_4 * h2_4 + h3_4 * h3_4 + h4_4 * h4_4 + h5_4 * h5_4
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 4, best_j)
    _cn = h0_5 * h0_5 + h1_5 * h1_5 + h2_5 * h2_5 + h3_5 * h3_5 + h4_5 * h4_5 + h5_5 * h5_5
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 5, best_j)
    ev1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev1_0 = tl.where(best_j == 0, h0_0, ev1_0)
    ev1_0 = tl.where(best_j == 1, h0_1, ev1_0)
    ev1_0 = tl.where(best_j == 2, h0_2, ev1_0)
    ev1_0 = tl.where(best_j == 3, h0_3, ev1_0)
    ev1_0 = tl.where(best_j == 4, h0_4, ev1_0)
    ev1_0 = tl.where(best_j == 5, h0_5, ev1_0)
    ev1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev1_1 = tl.where(best_j == 0, h1_0, ev1_1)
    ev1_1 = tl.where(best_j == 1, h1_1, ev1_1)
    ev1_1 = tl.where(best_j == 2, h1_2, ev1_1)
    ev1_1 = tl.where(best_j == 3, h1_3, ev1_1)
    ev1_1 = tl.where(best_j == 4, h1_4, ev1_1)
    ev1_1 = tl.where(best_j == 5, h1_5, ev1_1)
    ev1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev1_2 = tl.where(best_j == 0, h2_0, ev1_2)
    ev1_2 = tl.where(best_j == 1, h2_1, ev1_2)
    ev1_2 = tl.where(best_j == 2, h2_2, ev1_2)
    ev1_2 = tl.where(best_j == 3, h2_3, ev1_2)
    ev1_2 = tl.where(best_j == 4, h2_4, ev1_2)
    ev1_2 = tl.where(best_j == 5, h2_5, ev1_2)
    ev1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev1_3 = tl.where(best_j == 0, h3_0, ev1_3)
    ev1_3 = tl.where(best_j == 1, h3_1, ev1_3)
    ev1_3 = tl.where(best_j == 2, h3_2, ev1_3)
    ev1_3 = tl.where(best_j == 3, h3_3, ev1_3)
    ev1_3 = tl.where(best_j == 4, h3_4, ev1_3)
    ev1_3 = tl.where(best_j == 5, h3_5, ev1_3)
    ev1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev1_4 = tl.where(best_j == 0, h4_0, ev1_4)
    ev1_4 = tl.where(best_j == 1, h4_1, ev1_4)
    ev1_4 = tl.where(best_j == 2, h4_2, ev1_4)
    ev1_4 = tl.where(best_j == 3, h4_3, ev1_4)
    ev1_4 = tl.where(best_j == 4, h4_4, ev1_4)
    ev1_4 = tl.where(best_j == 5, h4_5, ev1_4)
    ev1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev1_5 = tl.where(best_j == 0, h5_0, ev1_5)
    ev1_5 = tl.where(best_j == 1, h5_1, ev1_5)
    ev1_5 = tl.where(best_j == 2, h5_2, ev1_5)
    ev1_5 = tl.where(best_j == 3, h5_3, ev1_5)
    ev1_5 = tl.where(best_j == 4, h5_4, ev1_5)
    ev1_5 = tl.where(best_j == 5, h5_5, ev1_5)
    _vn = tl.sqrt(ev1_0 * ev1_0 + ev1_1 * ev1_1 + ev1_2 * ev1_2 + ev1_3 * ev1_3 + ev1_4 * ev1_4 + ev1_5 * ev1_5 + 1e-60)
    ev1_0 = ev1_0 / _vn
    ev1_1 = ev1_1 / _vn
    ev1_2 = ev1_2 / _vn
    ev1_3 = ev1_3 / _vn
    ev1_4 = ev1_4 / _vn
    ev1_5 = ev1_5 / _vn
    # Eigenvector 2
    lam = r2
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    h0_0 = mn0_0
    h0_1 = mn0_1
    h0_2 = mn0_2
    h0_3 = mn0_3
    h0_4 = mn0_4
    h0_5 = mn0_5
    h1_0 = mn1_0
    h1_1 = mn1_1
    h1_2 = mn1_2
    h1_3 = mn1_3
    h1_4 = mn1_4
    h1_5 = mn1_5
    h2_0 = mn2_0
    h2_1 = mn2_1
    h2_2 = mn2_2
    h2_3 = mn2_3
    h2_4 = mn2_4
    h2_5 = mn2_5
    h3_0 = mn3_0
    h3_1 = mn3_1
    h3_2 = mn3_2
    h3_3 = mn3_3
    h3_4 = mn3_4
    h3_5 = mn3_5
    h4_0 = mn4_0
    h4_1 = mn4_1
    h4_2 = mn4_2
    h4_3 = mn4_3
    h4_4 = mn4_4
    h4_5 = mn4_5
    h5_0 = mn5_0
    h5_1 = mn5_1
    h5_2 = mn5_2
    h5_3 = mn5_3
    h5_4 = mn5_4
    h5_5 = mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)
    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0
    _cn = h0_0 * h0_0 + h1_0 * h1_0 + h2_0 * h2_0 + h3_0 * h3_0 + h4_0 * h4_0 + h5_0 * h5_0
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 0, best_j)
    _cn = h0_1 * h0_1 + h1_1 * h1_1 + h2_1 * h2_1 + h3_1 * h3_1 + h4_1 * h4_1 + h5_1 * h5_1
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 1, best_j)
    _cn = h0_2 * h0_2 + h1_2 * h1_2 + h2_2 * h2_2 + h3_2 * h3_2 + h4_2 * h4_2 + h5_2 * h5_2
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 2, best_j)
    _cn = h0_3 * h0_3 + h1_3 * h1_3 + h2_3 * h2_3 + h3_3 * h3_3 + h4_3 * h4_3 + h5_3 * h5_3
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 3, best_j)
    _cn = h0_4 * h0_4 + h1_4 * h1_4 + h2_4 * h2_4 + h3_4 * h3_4 + h4_4 * h4_4 + h5_4 * h5_4
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 4, best_j)
    _cn = h0_5 * h0_5 + h1_5 * h1_5 + h2_5 * h2_5 + h3_5 * h3_5 + h4_5 * h4_5 + h5_5 * h5_5
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 5, best_j)
    ev2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev2_0 = tl.where(best_j == 0, h0_0, ev2_0)
    ev2_0 = tl.where(best_j == 1, h0_1, ev2_0)
    ev2_0 = tl.where(best_j == 2, h0_2, ev2_0)
    ev2_0 = tl.where(best_j == 3, h0_3, ev2_0)
    ev2_0 = tl.where(best_j == 4, h0_4, ev2_0)
    ev2_0 = tl.where(best_j == 5, h0_5, ev2_0)
    ev2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev2_1 = tl.where(best_j == 0, h1_0, ev2_1)
    ev2_1 = tl.where(best_j == 1, h1_1, ev2_1)
    ev2_1 = tl.where(best_j == 2, h1_2, ev2_1)
    ev2_1 = tl.where(best_j == 3, h1_3, ev2_1)
    ev2_1 = tl.where(best_j == 4, h1_4, ev2_1)
    ev2_1 = tl.where(best_j == 5, h1_5, ev2_1)
    ev2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev2_2 = tl.where(best_j == 0, h2_0, ev2_2)
    ev2_2 = tl.where(best_j == 1, h2_1, ev2_2)
    ev2_2 = tl.where(best_j == 2, h2_2, ev2_2)
    ev2_2 = tl.where(best_j == 3, h2_3, ev2_2)
    ev2_2 = tl.where(best_j == 4, h2_4, ev2_2)
    ev2_2 = tl.where(best_j == 5, h2_5, ev2_2)
    ev2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev2_3 = tl.where(best_j == 0, h3_0, ev2_3)
    ev2_3 = tl.where(best_j == 1, h3_1, ev2_3)
    ev2_3 = tl.where(best_j == 2, h3_2, ev2_3)
    ev2_3 = tl.where(best_j == 3, h3_3, ev2_3)
    ev2_3 = tl.where(best_j == 4, h3_4, ev2_3)
    ev2_3 = tl.where(best_j == 5, h3_5, ev2_3)
    ev2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev2_4 = tl.where(best_j == 0, h4_0, ev2_4)
    ev2_4 = tl.where(best_j == 1, h4_1, ev2_4)
    ev2_4 = tl.where(best_j == 2, h4_2, ev2_4)
    ev2_4 = tl.where(best_j == 3, h4_3, ev2_4)
    ev2_4 = tl.where(best_j == 4, h4_4, ev2_4)
    ev2_4 = tl.where(best_j == 5, h4_5, ev2_4)
    ev2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev2_5 = tl.where(best_j == 0, h5_0, ev2_5)
    ev2_5 = tl.where(best_j == 1, h5_1, ev2_5)
    ev2_5 = tl.where(best_j == 2, h5_2, ev2_5)
    ev2_5 = tl.where(best_j == 3, h5_3, ev2_5)
    ev2_5 = tl.where(best_j == 4, h5_4, ev2_5)
    ev2_5 = tl.where(best_j == 5, h5_5, ev2_5)
    _vn = tl.sqrt(ev2_0 * ev2_0 + ev2_1 * ev2_1 + ev2_2 * ev2_2 + ev2_3 * ev2_3 + ev2_4 * ev2_4 + ev2_5 * ev2_5 + 1e-60)
    ev2_0 = ev2_0 / _vn
    ev2_1 = ev2_1 / _vn
    ev2_2 = ev2_2 / _vn
    ev2_3 = ev2_3 / _vn
    ev2_4 = ev2_4 / _vn
    ev2_5 = ev2_5 / _vn
    # Eigenvector 3
    lam = r3
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    h0_0 = mn0_0
    h0_1 = mn0_1
    h0_2 = mn0_2
    h0_3 = mn0_3
    h0_4 = mn0_4
    h0_5 = mn0_5
    h1_0 = mn1_0
    h1_1 = mn1_1
    h1_2 = mn1_2
    h1_3 = mn1_3
    h1_4 = mn1_4
    h1_5 = mn1_5
    h2_0 = mn2_0
    h2_1 = mn2_1
    h2_2 = mn2_2
    h2_3 = mn2_3
    h2_4 = mn2_4
    h2_5 = mn2_5
    h3_0 = mn3_0
    h3_1 = mn3_1
    h3_2 = mn3_2
    h3_3 = mn3_3
    h3_4 = mn3_4
    h3_5 = mn3_5
    h4_0 = mn4_0
    h4_1 = mn4_1
    h4_2 = mn4_2
    h4_3 = mn4_3
    h4_4 = mn4_4
    h4_5 = mn4_5
    h5_0 = mn5_0
    h5_1 = mn5_1
    h5_2 = mn5_2
    h5_3 = mn5_3
    h5_4 = mn5_4
    h5_5 = mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)
    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0
    _cn = h0_0 * h0_0 + h1_0 * h1_0 + h2_0 * h2_0 + h3_0 * h3_0 + h4_0 * h4_0 + h5_0 * h5_0
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 0, best_j)
    _cn = h0_1 * h0_1 + h1_1 * h1_1 + h2_1 * h2_1 + h3_1 * h3_1 + h4_1 * h4_1 + h5_1 * h5_1
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 1, best_j)
    _cn = h0_2 * h0_2 + h1_2 * h1_2 + h2_2 * h2_2 + h3_2 * h3_2 + h4_2 * h4_2 + h5_2 * h5_2
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 2, best_j)
    _cn = h0_3 * h0_3 + h1_3 * h1_3 + h2_3 * h2_3 + h3_3 * h3_3 + h4_3 * h4_3 + h5_3 * h5_3
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 3, best_j)
    _cn = h0_4 * h0_4 + h1_4 * h1_4 + h2_4 * h2_4 + h3_4 * h3_4 + h4_4 * h4_4 + h5_4 * h5_4
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 4, best_j)
    _cn = h0_5 * h0_5 + h1_5 * h1_5 + h2_5 * h2_5 + h3_5 * h3_5 + h4_5 * h4_5 + h5_5 * h5_5
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 5, best_j)
    ev3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev3_0 = tl.where(best_j == 0, h0_0, ev3_0)
    ev3_0 = tl.where(best_j == 1, h0_1, ev3_0)
    ev3_0 = tl.where(best_j == 2, h0_2, ev3_0)
    ev3_0 = tl.where(best_j == 3, h0_3, ev3_0)
    ev3_0 = tl.where(best_j == 4, h0_4, ev3_0)
    ev3_0 = tl.where(best_j == 5, h0_5, ev3_0)
    ev3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev3_1 = tl.where(best_j == 0, h1_0, ev3_1)
    ev3_1 = tl.where(best_j == 1, h1_1, ev3_1)
    ev3_1 = tl.where(best_j == 2, h1_2, ev3_1)
    ev3_1 = tl.where(best_j == 3, h1_3, ev3_1)
    ev3_1 = tl.where(best_j == 4, h1_4, ev3_1)
    ev3_1 = tl.where(best_j == 5, h1_5, ev3_1)
    ev3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev3_2 = tl.where(best_j == 0, h2_0, ev3_2)
    ev3_2 = tl.where(best_j == 1, h2_1, ev3_2)
    ev3_2 = tl.where(best_j == 2, h2_2, ev3_2)
    ev3_2 = tl.where(best_j == 3, h2_3, ev3_2)
    ev3_2 = tl.where(best_j == 4, h2_4, ev3_2)
    ev3_2 = tl.where(best_j == 5, h2_5, ev3_2)
    ev3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev3_3 = tl.where(best_j == 0, h3_0, ev3_3)
    ev3_3 = tl.where(best_j == 1, h3_1, ev3_3)
    ev3_3 = tl.where(best_j == 2, h3_2, ev3_3)
    ev3_3 = tl.where(best_j == 3, h3_3, ev3_3)
    ev3_3 = tl.where(best_j == 4, h3_4, ev3_3)
    ev3_3 = tl.where(best_j == 5, h3_5, ev3_3)
    ev3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev3_4 = tl.where(best_j == 0, h4_0, ev3_4)
    ev3_4 = tl.where(best_j == 1, h4_1, ev3_4)
    ev3_4 = tl.where(best_j == 2, h4_2, ev3_4)
    ev3_4 = tl.where(best_j == 3, h4_3, ev3_4)
    ev3_4 = tl.where(best_j == 4, h4_4, ev3_4)
    ev3_4 = tl.where(best_j == 5, h4_5, ev3_4)
    ev3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev3_5 = tl.where(best_j == 0, h5_0, ev3_5)
    ev3_5 = tl.where(best_j == 1, h5_1, ev3_5)
    ev3_5 = tl.where(best_j == 2, h5_2, ev3_5)
    ev3_5 = tl.where(best_j == 3, h5_3, ev3_5)
    ev3_5 = tl.where(best_j == 4, h5_4, ev3_5)
    ev3_5 = tl.where(best_j == 5, h5_5, ev3_5)
    _vn = tl.sqrt(ev3_0 * ev3_0 + ev3_1 * ev3_1 + ev3_2 * ev3_2 + ev3_3 * ev3_3 + ev3_4 * ev3_4 + ev3_5 * ev3_5 + 1e-60)
    ev3_0 = ev3_0 / _vn
    ev3_1 = ev3_1 / _vn
    ev3_2 = ev3_2 / _vn
    ev3_3 = ev3_3 / _vn
    ev3_4 = ev3_4 / _vn
    ev3_5 = ev3_5 / _vn
    # Eigenvector 4
    lam = r4
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    h0_0 = mn0_0
    h0_1 = mn0_1
    h0_2 = mn0_2
    h0_3 = mn0_3
    h0_4 = mn0_4
    h0_5 = mn0_5
    h1_0 = mn1_0
    h1_1 = mn1_1
    h1_2 = mn1_2
    h1_3 = mn1_3
    h1_4 = mn1_4
    h1_5 = mn1_5
    h2_0 = mn2_0
    h2_1 = mn2_1
    h2_2 = mn2_2
    h2_3 = mn2_3
    h2_4 = mn2_4
    h2_5 = mn2_5
    h3_0 = mn3_0
    h3_1 = mn3_1
    h3_2 = mn3_2
    h3_3 = mn3_3
    h3_4 = mn3_4
    h3_5 = mn3_5
    h4_0 = mn4_0
    h4_1 = mn4_1
    h4_2 = mn4_2
    h4_3 = mn4_3
    h4_4 = mn4_4
    h4_5 = mn4_5
    h5_0 = mn5_0
    h5_1 = mn5_1
    h5_2 = mn5_2
    h5_3 = mn5_3
    h5_4 = mn5_4
    h5_5 = mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)
    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0
    _cn = h0_0 * h0_0 + h1_0 * h1_0 + h2_0 * h2_0 + h3_0 * h3_0 + h4_0 * h4_0 + h5_0 * h5_0
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 0, best_j)
    _cn = h0_1 * h0_1 + h1_1 * h1_1 + h2_1 * h2_1 + h3_1 * h3_1 + h4_1 * h4_1 + h5_1 * h5_1
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 1, best_j)
    _cn = h0_2 * h0_2 + h1_2 * h1_2 + h2_2 * h2_2 + h3_2 * h3_2 + h4_2 * h4_2 + h5_2 * h5_2
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 2, best_j)
    _cn = h0_3 * h0_3 + h1_3 * h1_3 + h2_3 * h2_3 + h3_3 * h3_3 + h4_3 * h4_3 + h5_3 * h5_3
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 3, best_j)
    _cn = h0_4 * h0_4 + h1_4 * h1_4 + h2_4 * h2_4 + h3_4 * h3_4 + h4_4 * h4_4 + h5_4 * h5_4
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 4, best_j)
    _cn = h0_5 * h0_5 + h1_5 * h1_5 + h2_5 * h2_5 + h3_5 * h3_5 + h4_5 * h4_5 + h5_5 * h5_5
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 5, best_j)
    ev4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev4_0 = tl.where(best_j == 0, h0_0, ev4_0)
    ev4_0 = tl.where(best_j == 1, h0_1, ev4_0)
    ev4_0 = tl.where(best_j == 2, h0_2, ev4_0)
    ev4_0 = tl.where(best_j == 3, h0_3, ev4_0)
    ev4_0 = tl.where(best_j == 4, h0_4, ev4_0)
    ev4_0 = tl.where(best_j == 5, h0_5, ev4_0)
    ev4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev4_1 = tl.where(best_j == 0, h1_0, ev4_1)
    ev4_1 = tl.where(best_j == 1, h1_1, ev4_1)
    ev4_1 = tl.where(best_j == 2, h1_2, ev4_1)
    ev4_1 = tl.where(best_j == 3, h1_3, ev4_1)
    ev4_1 = tl.where(best_j == 4, h1_4, ev4_1)
    ev4_1 = tl.where(best_j == 5, h1_5, ev4_1)
    ev4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev4_2 = tl.where(best_j == 0, h2_0, ev4_2)
    ev4_2 = tl.where(best_j == 1, h2_1, ev4_2)
    ev4_2 = tl.where(best_j == 2, h2_2, ev4_2)
    ev4_2 = tl.where(best_j == 3, h2_3, ev4_2)
    ev4_2 = tl.where(best_j == 4, h2_4, ev4_2)
    ev4_2 = tl.where(best_j == 5, h2_5, ev4_2)
    ev4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev4_3 = tl.where(best_j == 0, h3_0, ev4_3)
    ev4_3 = tl.where(best_j == 1, h3_1, ev4_3)
    ev4_3 = tl.where(best_j == 2, h3_2, ev4_3)
    ev4_3 = tl.where(best_j == 3, h3_3, ev4_3)
    ev4_3 = tl.where(best_j == 4, h3_4, ev4_3)
    ev4_3 = tl.where(best_j == 5, h3_5, ev4_3)
    ev4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev4_4 = tl.where(best_j == 0, h4_0, ev4_4)
    ev4_4 = tl.where(best_j == 1, h4_1, ev4_4)
    ev4_4 = tl.where(best_j == 2, h4_2, ev4_4)
    ev4_4 = tl.where(best_j == 3, h4_3, ev4_4)
    ev4_4 = tl.where(best_j == 4, h4_4, ev4_4)
    ev4_4 = tl.where(best_j == 5, h4_5, ev4_4)
    ev4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev4_5 = tl.where(best_j == 0, h5_0, ev4_5)
    ev4_5 = tl.where(best_j == 1, h5_1, ev4_5)
    ev4_5 = tl.where(best_j == 2, h5_2, ev4_5)
    ev4_5 = tl.where(best_j == 3, h5_3, ev4_5)
    ev4_5 = tl.where(best_j == 4, h5_4, ev4_5)
    ev4_5 = tl.where(best_j == 5, h5_5, ev4_5)
    _vn = tl.sqrt(ev4_0 * ev4_0 + ev4_1 * ev4_1 + ev4_2 * ev4_2 + ev4_3 * ev4_3 + ev4_4 * ev4_4 + ev4_5 * ev4_5 + 1e-60)
    ev4_0 = ev4_0 / _vn
    ev4_1 = ev4_1 / _vn
    ev4_2 = ev4_2 / _vn
    ev4_3 = ev4_3 / _vn
    ev4_4 = ev4_4 / _vn
    ev4_5 = ev4_5 / _vn
    # Eigenvector 5
    lam = r5
    m0_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m0_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m1_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m2_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m3_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m4_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    m5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c6
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c6
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c6
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c6
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c6
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c6
    h0_0 = mn0_0
    h0_1 = mn0_1
    h0_2 = mn0_2
    h0_3 = mn0_3
    h0_4 = mn0_4
    h0_5 = mn0_5
    h1_0 = mn1_0
    h1_1 = mn1_1
    h1_2 = mn1_2
    h1_3 = mn1_3
    h1_4 = mn1_4
    h1_5 = mn1_5
    h2_0 = mn2_0
    h2_1 = mn2_1
    h2_2 = mn2_2
    h2_3 = mn2_3
    h2_4 = mn2_4
    h2_5 = mn2_5
    h3_0 = mn3_0
    h3_1 = mn3_1
    h3_2 = mn3_2
    h3_3 = mn3_3
    h3_4 = mn3_4
    h3_5 = mn3_5
    h4_0 = mn4_0
    h4_1 = mn4_1
    h4_2 = mn4_2
    h4_3 = mn4_3
    h4_4 = mn4_4
    h4_5 = mn4_5
    h5_0 = mn5_0
    h5_1 = mn5_1
    h5_2 = mn5_2
    h5_3 = mn5_3
    h5_4 = mn5_4
    h5_5 = mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c5
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c5
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c5
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c5
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c5
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c5
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c4
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c4
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c4
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c4
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c4
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c4
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c3
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c3
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c3
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c3
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c3
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c3
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c2
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c2
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c2
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c2
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c2
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c2
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    mn0_0 = a0_0 * m0_0 + a0_1 * m1_0 + a0_2 * m2_0 + a0_3 * m3_0 + a0_4 * m4_0 + a0_5 * m5_0 + c1
    mn0_1 = a0_0 * m0_1 + a0_1 * m1_1 + a0_2 * m2_1 + a0_3 * m3_1 + a0_4 * m4_1 + a0_5 * m5_1
    mn0_2 = a0_0 * m0_2 + a0_1 * m1_2 + a0_2 * m2_2 + a0_3 * m3_2 + a0_4 * m4_2 + a0_5 * m5_2
    mn0_3 = a0_0 * m0_3 + a0_1 * m1_3 + a0_2 * m2_3 + a0_3 * m3_3 + a0_4 * m4_3 + a0_5 * m5_3
    mn0_4 = a0_0 * m0_4 + a0_1 * m1_4 + a0_2 * m2_4 + a0_3 * m3_4 + a0_4 * m4_4 + a0_5 * m5_4
    mn0_5 = a0_0 * m0_5 + a0_1 * m1_5 + a0_2 * m2_5 + a0_3 * m3_5 + a0_4 * m4_5 + a0_5 * m5_5
    mn1_0 = a1_0 * m0_0 + a1_1 * m1_0 + a1_2 * m2_0 + a1_3 * m3_0 + a1_4 * m4_0 + a1_5 * m5_0
    mn1_1 = a1_0 * m0_1 + a1_1 * m1_1 + a1_2 * m2_1 + a1_3 * m3_1 + a1_4 * m4_1 + a1_5 * m5_1 + c1
    mn1_2 = a1_0 * m0_2 + a1_1 * m1_2 + a1_2 * m2_2 + a1_3 * m3_2 + a1_4 * m4_2 + a1_5 * m5_2
    mn1_3 = a1_0 * m0_3 + a1_1 * m1_3 + a1_2 * m2_3 + a1_3 * m3_3 + a1_4 * m4_3 + a1_5 * m5_3
    mn1_4 = a1_0 * m0_4 + a1_1 * m1_4 + a1_2 * m2_4 + a1_3 * m3_4 + a1_4 * m4_4 + a1_5 * m5_4
    mn1_5 = a1_0 * m0_5 + a1_1 * m1_5 + a1_2 * m2_5 + a1_3 * m3_5 + a1_4 * m4_5 + a1_5 * m5_5
    mn2_0 = a2_0 * m0_0 + a2_1 * m1_0 + a2_2 * m2_0 + a2_3 * m3_0 + a2_4 * m4_0 + a2_5 * m5_0
    mn2_1 = a2_0 * m0_1 + a2_1 * m1_1 + a2_2 * m2_1 + a2_3 * m3_1 + a2_4 * m4_1 + a2_5 * m5_1
    mn2_2 = a2_0 * m0_2 + a2_1 * m1_2 + a2_2 * m2_2 + a2_3 * m3_2 + a2_4 * m4_2 + a2_5 * m5_2 + c1
    mn2_3 = a2_0 * m0_3 + a2_1 * m1_3 + a2_2 * m2_3 + a2_3 * m3_3 + a2_4 * m4_3 + a2_5 * m5_3
    mn2_4 = a2_0 * m0_4 + a2_1 * m1_4 + a2_2 * m2_4 + a2_3 * m3_4 + a2_4 * m4_4 + a2_5 * m5_4
    mn2_5 = a2_0 * m0_5 + a2_1 * m1_5 + a2_2 * m2_5 + a2_3 * m3_5 + a2_4 * m4_5 + a2_5 * m5_5
    mn3_0 = a3_0 * m0_0 + a3_1 * m1_0 + a3_2 * m2_0 + a3_3 * m3_0 + a3_4 * m4_0 + a3_5 * m5_0
    mn3_1 = a3_0 * m0_1 + a3_1 * m1_1 + a3_2 * m2_1 + a3_3 * m3_1 + a3_4 * m4_1 + a3_5 * m5_1
    mn3_2 = a3_0 * m0_2 + a3_1 * m1_2 + a3_2 * m2_2 + a3_3 * m3_2 + a3_4 * m4_2 + a3_5 * m5_2
    mn3_3 = a3_0 * m0_3 + a3_1 * m1_3 + a3_2 * m2_3 + a3_3 * m3_3 + a3_4 * m4_3 + a3_5 * m5_3 + c1
    mn3_4 = a3_0 * m0_4 + a3_1 * m1_4 + a3_2 * m2_4 + a3_3 * m3_4 + a3_4 * m4_4 + a3_5 * m5_4
    mn3_5 = a3_0 * m0_5 + a3_1 * m1_5 + a3_2 * m2_5 + a3_3 * m3_5 + a3_4 * m4_5 + a3_5 * m5_5
    mn4_0 = a4_0 * m0_0 + a4_1 * m1_0 + a4_2 * m2_0 + a4_3 * m3_0 + a4_4 * m4_0 + a4_5 * m5_0
    mn4_1 = a4_0 * m0_1 + a4_1 * m1_1 + a4_2 * m2_1 + a4_3 * m3_1 + a4_4 * m4_1 + a4_5 * m5_1
    mn4_2 = a4_0 * m0_2 + a4_1 * m1_2 + a4_2 * m2_2 + a4_3 * m3_2 + a4_4 * m4_2 + a4_5 * m5_2
    mn4_3 = a4_0 * m0_3 + a4_1 * m1_3 + a4_2 * m2_3 + a4_3 * m3_3 + a4_4 * m4_3 + a4_5 * m5_3
    mn4_4 = a4_0 * m0_4 + a4_1 * m1_4 + a4_2 * m2_4 + a4_3 * m3_4 + a4_4 * m4_4 + a4_5 * m5_4 + c1
    mn4_5 = a4_0 * m0_5 + a4_1 * m1_5 + a4_2 * m2_5 + a4_3 * m3_5 + a4_4 * m4_5 + a4_5 * m5_5
    mn5_0 = a5_0 * m0_0 + a5_1 * m1_0 + a5_2 * m2_0 + a5_3 * m3_0 + a5_4 * m4_0 + a5_5 * m5_0
    mn5_1 = a5_0 * m0_1 + a5_1 * m1_1 + a5_2 * m2_1 + a5_3 * m3_1 + a5_4 * m4_1 + a5_5 * m5_1
    mn5_2 = a5_0 * m0_2 + a5_1 * m1_2 + a5_2 * m2_2 + a5_3 * m3_2 + a5_4 * m4_2 + a5_5 * m5_2
    mn5_3 = a5_0 * m0_3 + a5_1 * m1_3 + a5_2 * m2_3 + a5_3 * m3_3 + a5_4 * m4_3 + a5_5 * m5_3
    mn5_4 = a5_0 * m0_4 + a5_1 * m1_4 + a5_2 * m2_4 + a5_3 * m3_4 + a5_4 * m4_4 + a5_5 * m5_4
    mn5_5 = a5_0 * m0_5 + a5_1 * m1_5 + a5_2 * m2_5 + a5_3 * m3_5 + a5_4 * m4_5 + a5_5 * m5_5 + c1
    h0_0 = h0_0 * lam + mn0_0
    h0_1 = h0_1 * lam + mn0_1
    h0_2 = h0_2 * lam + mn0_2
    h0_3 = h0_3 * lam + mn0_3
    h0_4 = h0_4 * lam + mn0_4
    h0_5 = h0_5 * lam + mn0_5
    h1_0 = h1_0 * lam + mn1_0
    h1_1 = h1_1 * lam + mn1_1
    h1_2 = h1_2 * lam + mn1_2
    h1_3 = h1_3 * lam + mn1_3
    h1_4 = h1_4 * lam + mn1_4
    h1_5 = h1_5 * lam + mn1_5
    h2_0 = h2_0 * lam + mn2_0
    h2_1 = h2_1 * lam + mn2_1
    h2_2 = h2_2 * lam + mn2_2
    h2_3 = h2_3 * lam + mn2_3
    h2_4 = h2_4 * lam + mn2_4
    h2_5 = h2_5 * lam + mn2_5
    h3_0 = h3_0 * lam + mn3_0
    h3_1 = h3_1 * lam + mn3_1
    h3_2 = h3_2 * lam + mn3_2
    h3_3 = h3_3 * lam + mn3_3
    h3_4 = h3_4 * lam + mn3_4
    h3_5 = h3_5 * lam + mn3_5
    h4_0 = h4_0 * lam + mn4_0
    h4_1 = h4_1 * lam + mn4_1
    h4_2 = h4_2 * lam + mn4_2
    h4_3 = h4_3 * lam + mn4_3
    h4_4 = h4_4 * lam + mn4_4
    h4_5 = h4_5 * lam + mn4_5
    h5_0 = h5_0 * lam + mn5_0
    h5_1 = h5_1 * lam + mn5_1
    h5_2 = h5_2 * lam + mn5_2
    h5_3 = h5_3 * lam + mn5_3
    h5_4 = h5_4 * lam + mn5_4
    h5_5 = h5_5 * lam + mn5_5
    m0_0 = mn0_0
    m0_1 = mn0_1
    m0_2 = mn0_2
    m0_3 = mn0_3
    m0_4 = mn0_4
    m0_5 = mn0_5
    m1_0 = mn1_0
    m1_1 = mn1_1
    m1_2 = mn1_2
    m1_3 = mn1_3
    m1_4 = mn1_4
    m1_5 = mn1_5
    m2_0 = mn2_0
    m2_1 = mn2_1
    m2_2 = mn2_2
    m2_3 = mn2_3
    m2_4 = mn2_4
    m2_5 = mn2_5
    m3_0 = mn3_0
    m3_1 = mn3_1
    m3_2 = mn3_2
    m3_3 = mn3_3
    m3_4 = mn3_4
    m3_5 = mn3_5
    m4_0 = mn4_0
    m4_1 = mn4_1
    m4_2 = mn4_2
    m4_3 = mn4_3
    m4_4 = mn4_4
    m4_5 = mn4_5
    m5_0 = mn5_0
    m5_1 = mn5_1
    m5_2 = mn5_2
    m5_3 = mn5_3
    m5_4 = mn5_4
    m5_5 = mn5_5
    best_j = tl.zeros((BLOCK_B,), dtype=tl.int32)
    best_n2 = tl.zeros((BLOCK_B,), dtype=tl.float64) - 1.0
    _cn = h0_0 * h0_0 + h1_0 * h1_0 + h2_0 * h2_0 + h3_0 * h3_0 + h4_0 * h4_0 + h5_0 * h5_0
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 0, best_j)
    _cn = h0_1 * h0_1 + h1_1 * h1_1 + h2_1 * h2_1 + h3_1 * h3_1 + h4_1 * h4_1 + h5_1 * h5_1
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 1, best_j)
    _cn = h0_2 * h0_2 + h1_2 * h1_2 + h2_2 * h2_2 + h3_2 * h3_2 + h4_2 * h4_2 + h5_2 * h5_2
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 2, best_j)
    _cn = h0_3 * h0_3 + h1_3 * h1_3 + h2_3 * h2_3 + h3_3 * h3_3 + h4_3 * h4_3 + h5_3 * h5_3
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 3, best_j)
    _cn = h0_4 * h0_4 + h1_4 * h1_4 + h2_4 * h2_4 + h3_4 * h3_4 + h4_4 * h4_4 + h5_4 * h5_4
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 4, best_j)
    _cn = h0_5 * h0_5 + h1_5 * h1_5 + h2_5 * h2_5 + h3_5 * h3_5 + h4_5 * h4_5 + h5_5 * h5_5
    _better = _cn > best_n2
    best_n2 = tl.where(_better, _cn, best_n2)
    best_j = tl.where(_better, 5, best_j)
    ev5_0 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev5_0 = tl.where(best_j == 0, h0_0, ev5_0)
    ev5_0 = tl.where(best_j == 1, h0_1, ev5_0)
    ev5_0 = tl.where(best_j == 2, h0_2, ev5_0)
    ev5_0 = tl.where(best_j == 3, h0_3, ev5_0)
    ev5_0 = tl.where(best_j == 4, h0_4, ev5_0)
    ev5_0 = tl.where(best_j == 5, h0_5, ev5_0)
    ev5_1 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev5_1 = tl.where(best_j == 0, h1_0, ev5_1)
    ev5_1 = tl.where(best_j == 1, h1_1, ev5_1)
    ev5_1 = tl.where(best_j == 2, h1_2, ev5_1)
    ev5_1 = tl.where(best_j == 3, h1_3, ev5_1)
    ev5_1 = tl.where(best_j == 4, h1_4, ev5_1)
    ev5_1 = tl.where(best_j == 5, h1_5, ev5_1)
    ev5_2 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev5_2 = tl.where(best_j == 0, h2_0, ev5_2)
    ev5_2 = tl.where(best_j == 1, h2_1, ev5_2)
    ev5_2 = tl.where(best_j == 2, h2_2, ev5_2)
    ev5_2 = tl.where(best_j == 3, h2_3, ev5_2)
    ev5_2 = tl.where(best_j == 4, h2_4, ev5_2)
    ev5_2 = tl.where(best_j == 5, h2_5, ev5_2)
    ev5_3 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev5_3 = tl.where(best_j == 0, h3_0, ev5_3)
    ev5_3 = tl.where(best_j == 1, h3_1, ev5_3)
    ev5_3 = tl.where(best_j == 2, h3_2, ev5_3)
    ev5_3 = tl.where(best_j == 3, h3_3, ev5_3)
    ev5_3 = tl.where(best_j == 4, h3_4, ev5_3)
    ev5_3 = tl.where(best_j == 5, h3_5, ev5_3)
    ev5_4 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev5_4 = tl.where(best_j == 0, h4_0, ev5_4)
    ev5_4 = tl.where(best_j == 1, h4_1, ev5_4)
    ev5_4 = tl.where(best_j == 2, h4_2, ev5_4)
    ev5_4 = tl.where(best_j == 3, h4_3, ev5_4)
    ev5_4 = tl.where(best_j == 4, h4_4, ev5_4)
    ev5_4 = tl.where(best_j == 5, h4_5, ev5_4)
    ev5_5 = tl.zeros((BLOCK_B,), dtype=tl.float64)
    ev5_5 = tl.where(best_j == 0, h5_0, ev5_5)
    ev5_5 = tl.where(best_j == 1, h5_1, ev5_5)
    ev5_5 = tl.where(best_j == 2, h5_2, ev5_5)
    ev5_5 = tl.where(best_j == 3, h5_3, ev5_5)
    ev5_5 = tl.where(best_j == 4, h5_4, ev5_5)
    ev5_5 = tl.where(best_j == 5, h5_5, ev5_5)
    _vn = tl.sqrt(ev5_0 * ev5_0 + ev5_1 * ev5_1 + ev5_2 * ev5_2 + ev5_3 * ev5_3 + ev5_4 * ev5_4 + ev5_5 * ev5_5 + 1e-60)
    ev5_0 = ev5_0 / _vn
    ev5_1 = ev5_1 / _vn
    ev5_2 = ev5_2 / _vn
    ev5_3 = ev5_3 / _vn
    ev5_4 = ev5_4 / _vn
    ev5_5 = ev5_5 / _vn
    # Phase 4: NS
    v0_0 = ev0_0.to(tl.float32)
    v0_1 = ev1_0.to(tl.float32)
    v0_2 = ev2_0.to(tl.float32)
    v0_3 = ev3_0.to(tl.float32)
    v0_4 = ev4_0.to(tl.float32)
    v0_5 = ev5_0.to(tl.float32)
    v1_0 = ev0_1.to(tl.float32)
    v1_1 = ev1_1.to(tl.float32)
    v1_2 = ev2_1.to(tl.float32)
    v1_3 = ev3_1.to(tl.float32)
    v1_4 = ev4_1.to(tl.float32)
    v1_5 = ev5_1.to(tl.float32)
    v2_0 = ev0_2.to(tl.float32)
    v2_1 = ev1_2.to(tl.float32)
    v2_2 = ev2_2.to(tl.float32)
    v2_3 = ev3_2.to(tl.float32)
    v2_4 = ev4_2.to(tl.float32)
    v2_5 = ev5_2.to(tl.float32)
    v3_0 = ev0_3.to(tl.float32)
    v3_1 = ev1_3.to(tl.float32)
    v3_2 = ev2_3.to(tl.float32)
    v3_3 = ev3_3.to(tl.float32)
    v3_4 = ev4_3.to(tl.float32)
    v3_5 = ev5_3.to(tl.float32)
    v4_0 = ev0_4.to(tl.float32)
    v4_1 = ev1_4.to(tl.float32)
    v4_2 = ev2_4.to(tl.float32)
    v4_3 = ev3_4.to(tl.float32)
    v4_4 = ev4_4.to(tl.float32)
    v4_5 = ev5_4.to(tl.float32)
    v5_0 = ev0_5.to(tl.float32)
    v5_1 = ev1_5.to(tl.float32)
    v5_2 = ev2_5.to(tl.float32)
    v5_3 = ev3_5.to(tl.float32)
    v5_4 = ev4_5.to(tl.float32)
    v5_5 = ev5_5.to(tl.float32)
    # NS iter
    y0_0 = v0_0 * v0_0 + v1_0 * v1_0 + v2_0 * v2_0 + v3_0 * v3_0 + v4_0 * v4_0 + v5_0 * v5_0
    y0_1 = v0_0 * v0_1 + v1_0 * v1_1 + v2_0 * v2_1 + v3_0 * v3_1 + v4_0 * v4_1 + v5_0 * v5_1
    y0_2 = v0_0 * v0_2 + v1_0 * v1_2 + v2_0 * v2_2 + v3_0 * v3_2 + v4_0 * v4_2 + v5_0 * v5_2
    y0_3 = v0_0 * v0_3 + v1_0 * v1_3 + v2_0 * v2_3 + v3_0 * v3_3 + v4_0 * v4_3 + v5_0 * v5_3
    y0_4 = v0_0 * v0_4 + v1_0 * v1_4 + v2_0 * v2_4 + v3_0 * v3_4 + v4_0 * v4_4 + v5_0 * v5_4
    y0_5 = v0_0 * v0_5 + v1_0 * v1_5 + v2_0 * v2_5 + v3_0 * v3_5 + v4_0 * v4_5 + v5_0 * v5_5
    y1_0 = v0_1 * v0_0 + v1_1 * v1_0 + v2_1 * v2_0 + v3_1 * v3_0 + v4_1 * v4_0 + v5_1 * v5_0
    y1_1 = v0_1 * v0_1 + v1_1 * v1_1 + v2_1 * v2_1 + v3_1 * v3_1 + v4_1 * v4_1 + v5_1 * v5_1
    y1_2 = v0_1 * v0_2 + v1_1 * v1_2 + v2_1 * v2_2 + v3_1 * v3_2 + v4_1 * v4_2 + v5_1 * v5_2
    y1_3 = v0_1 * v0_3 + v1_1 * v1_3 + v2_1 * v2_3 + v3_1 * v3_3 + v4_1 * v4_3 + v5_1 * v5_3
    y1_4 = v0_1 * v0_4 + v1_1 * v1_4 + v2_1 * v2_4 + v3_1 * v3_4 + v4_1 * v4_4 + v5_1 * v5_4
    y1_5 = v0_1 * v0_5 + v1_1 * v1_5 + v2_1 * v2_5 + v3_1 * v3_5 + v4_1 * v4_5 + v5_1 * v5_5
    y2_0 = v0_2 * v0_0 + v1_2 * v1_0 + v2_2 * v2_0 + v3_2 * v3_0 + v4_2 * v4_0 + v5_2 * v5_0
    y2_1 = v0_2 * v0_1 + v1_2 * v1_1 + v2_2 * v2_1 + v3_2 * v3_1 + v4_2 * v4_1 + v5_2 * v5_1
    y2_2 = v0_2 * v0_2 + v1_2 * v1_2 + v2_2 * v2_2 + v3_2 * v3_2 + v4_2 * v4_2 + v5_2 * v5_2
    y2_3 = v0_2 * v0_3 + v1_2 * v1_3 + v2_2 * v2_3 + v3_2 * v3_3 + v4_2 * v4_3 + v5_2 * v5_3
    y2_4 = v0_2 * v0_4 + v1_2 * v1_4 + v2_2 * v2_4 + v3_2 * v3_4 + v4_2 * v4_4 + v5_2 * v5_4
    y2_5 = v0_2 * v0_5 + v1_2 * v1_5 + v2_2 * v2_5 + v3_2 * v3_5 + v4_2 * v4_5 + v5_2 * v5_5
    y3_0 = v0_3 * v0_0 + v1_3 * v1_0 + v2_3 * v2_0 + v3_3 * v3_0 + v4_3 * v4_0 + v5_3 * v5_0
    y3_1 = v0_3 * v0_1 + v1_3 * v1_1 + v2_3 * v2_1 + v3_3 * v3_1 + v4_3 * v4_1 + v5_3 * v5_1
    y3_2 = v0_3 * v0_2 + v1_3 * v1_2 + v2_3 * v2_2 + v3_3 * v3_2 + v4_3 * v4_2 + v5_3 * v5_2
    y3_3 = v0_3 * v0_3 + v1_3 * v1_3 + v2_3 * v2_3 + v3_3 * v3_3 + v4_3 * v4_3 + v5_3 * v5_3
    y3_4 = v0_3 * v0_4 + v1_3 * v1_4 + v2_3 * v2_4 + v3_3 * v3_4 + v4_3 * v4_4 + v5_3 * v5_4
    y3_5 = v0_3 * v0_5 + v1_3 * v1_5 + v2_3 * v2_5 + v3_3 * v3_5 + v4_3 * v4_5 + v5_3 * v5_5
    y4_0 = v0_4 * v0_0 + v1_4 * v1_0 + v2_4 * v2_0 + v3_4 * v3_0 + v4_4 * v4_0 + v5_4 * v5_0
    y4_1 = v0_4 * v0_1 + v1_4 * v1_1 + v2_4 * v2_1 + v3_4 * v3_1 + v4_4 * v4_1 + v5_4 * v5_1
    y4_2 = v0_4 * v0_2 + v1_4 * v1_2 + v2_4 * v2_2 + v3_4 * v3_2 + v4_4 * v4_2 + v5_4 * v5_2
    y4_3 = v0_4 * v0_3 + v1_4 * v1_3 + v2_4 * v2_3 + v3_4 * v3_3 + v4_4 * v4_3 + v5_4 * v5_3
    y4_4 = v0_4 * v0_4 + v1_4 * v1_4 + v2_4 * v2_4 + v3_4 * v3_4 + v4_4 * v4_4 + v5_4 * v5_4
    y4_5 = v0_4 * v0_5 + v1_4 * v1_5 + v2_4 * v2_5 + v3_4 * v3_5 + v4_4 * v4_5 + v5_4 * v5_5
    y5_0 = v0_5 * v0_0 + v1_5 * v1_0 + v2_5 * v2_0 + v3_5 * v3_0 + v4_5 * v4_0 + v5_5 * v5_0
    y5_1 = v0_5 * v0_1 + v1_5 * v1_1 + v2_5 * v2_1 + v3_5 * v3_1 + v4_5 * v4_1 + v5_5 * v5_1
    y5_2 = v0_5 * v0_2 + v1_5 * v1_2 + v2_5 * v2_2 + v3_5 * v3_2 + v4_5 * v4_2 + v5_5 * v5_2
    y5_3 = v0_5 * v0_3 + v1_5 * v1_3 + v2_5 * v2_3 + v3_5 * v3_3 + v4_5 * v4_3 + v5_5 * v5_3
    y5_4 = v0_5 * v0_4 + v1_5 * v1_4 + v2_5 * v2_4 + v3_5 * v3_4 + v4_5 * v4_4 + v5_5 * v5_4
    y5_5 = v0_5 * v0_5 + v1_5 * v1_5 + v2_5 * v2_5 + v3_5 * v3_5 + v4_5 * v4_5 + v5_5 * v5_5
    t0_0 = 3.0 - y0_0
    t0_1 = -y0_1
    t0_2 = -y0_2
    t0_3 = -y0_3
    t0_4 = -y0_4
    t0_5 = -y0_5
    t1_0 = -y1_0
    t1_1 = 3.0 - y1_1
    t1_2 = -y1_2
    t1_3 = -y1_3
    t1_4 = -y1_4
    t1_5 = -y1_5
    t2_0 = -y2_0
    t2_1 = -y2_1
    t2_2 = 3.0 - y2_2
    t2_3 = -y2_3
    t2_4 = -y2_4
    t2_5 = -y2_5
    t3_0 = -y3_0
    t3_1 = -y3_1
    t3_2 = -y3_2
    t3_3 = 3.0 - y3_3
    t3_4 = -y3_4
    t3_5 = -y3_5
    t4_0 = -y4_0
    t4_1 = -y4_1
    t4_2 = -y4_2
    t4_3 = -y4_3
    t4_4 = 3.0 - y4_4
    t4_5 = -y4_5
    t5_0 = -y5_0
    t5_1 = -y5_1
    t5_2 = -y5_2
    t5_3 = -y5_3
    t5_4 = -y5_4
    t5_5 = 3.0 - y5_5
    vn0_0 = 0.5 * (v0_0 * t0_0 + v0_1 * t1_0 + v0_2 * t2_0 + v0_3 * t3_0 + v0_4 * t4_0 + v0_5 * t5_0)
    vn0_1 = 0.5 * (v0_0 * t0_1 + v0_1 * t1_1 + v0_2 * t2_1 + v0_3 * t3_1 + v0_4 * t4_1 + v0_5 * t5_1)
    vn0_2 = 0.5 * (v0_0 * t0_2 + v0_1 * t1_2 + v0_2 * t2_2 + v0_3 * t3_2 + v0_4 * t4_2 + v0_5 * t5_2)
    vn0_3 = 0.5 * (v0_0 * t0_3 + v0_1 * t1_3 + v0_2 * t2_3 + v0_3 * t3_3 + v0_4 * t4_3 + v0_5 * t5_3)
    vn0_4 = 0.5 * (v0_0 * t0_4 + v0_1 * t1_4 + v0_2 * t2_4 + v0_3 * t3_4 + v0_4 * t4_4 + v0_5 * t5_4)
    vn0_5 = 0.5 * (v0_0 * t0_5 + v0_1 * t1_5 + v0_2 * t2_5 + v0_3 * t3_5 + v0_4 * t4_5 + v0_5 * t5_5)
    vn1_0 = 0.5 * (v1_0 * t0_0 + v1_1 * t1_0 + v1_2 * t2_0 + v1_3 * t3_0 + v1_4 * t4_0 + v1_5 * t5_0)
    vn1_1 = 0.5 * (v1_0 * t0_1 + v1_1 * t1_1 + v1_2 * t2_1 + v1_3 * t3_1 + v1_4 * t4_1 + v1_5 * t5_1)
    vn1_2 = 0.5 * (v1_0 * t0_2 + v1_1 * t1_2 + v1_2 * t2_2 + v1_3 * t3_2 + v1_4 * t4_2 + v1_5 * t5_2)
    vn1_3 = 0.5 * (v1_0 * t0_3 + v1_1 * t1_3 + v1_2 * t2_3 + v1_3 * t3_3 + v1_4 * t4_3 + v1_5 * t5_3)
    vn1_4 = 0.5 * (v1_0 * t0_4 + v1_1 * t1_4 + v1_2 * t2_4 + v1_3 * t3_4 + v1_4 * t4_4 + v1_5 * t5_4)
    vn1_5 = 0.5 * (v1_0 * t0_5 + v1_1 * t1_5 + v1_2 * t2_5 + v1_3 * t3_5 + v1_4 * t4_5 + v1_5 * t5_5)
    vn2_0 = 0.5 * (v2_0 * t0_0 + v2_1 * t1_0 + v2_2 * t2_0 + v2_3 * t3_0 + v2_4 * t4_0 + v2_5 * t5_0)
    vn2_1 = 0.5 * (v2_0 * t0_1 + v2_1 * t1_1 + v2_2 * t2_1 + v2_3 * t3_1 + v2_4 * t4_1 + v2_5 * t5_1)
    vn2_2 = 0.5 * (v2_0 * t0_2 + v2_1 * t1_2 + v2_2 * t2_2 + v2_3 * t3_2 + v2_4 * t4_2 + v2_5 * t5_2)
    vn2_3 = 0.5 * (v2_0 * t0_3 + v2_1 * t1_3 + v2_2 * t2_3 + v2_3 * t3_3 + v2_4 * t4_3 + v2_5 * t5_3)
    vn2_4 = 0.5 * (v2_0 * t0_4 + v2_1 * t1_4 + v2_2 * t2_4 + v2_3 * t3_4 + v2_4 * t4_4 + v2_5 * t5_4)
    vn2_5 = 0.5 * (v2_0 * t0_5 + v2_1 * t1_5 + v2_2 * t2_5 + v2_3 * t3_5 + v2_4 * t4_5 + v2_5 * t5_5)
    vn3_0 = 0.5 * (v3_0 * t0_0 + v3_1 * t1_0 + v3_2 * t2_0 + v3_3 * t3_0 + v3_4 * t4_0 + v3_5 * t5_0)
    vn3_1 = 0.5 * (v3_0 * t0_1 + v3_1 * t1_1 + v3_2 * t2_1 + v3_3 * t3_1 + v3_4 * t4_1 + v3_5 * t5_1)
    vn3_2 = 0.5 * (v3_0 * t0_2 + v3_1 * t1_2 + v3_2 * t2_2 + v3_3 * t3_2 + v3_4 * t4_2 + v3_5 * t5_2)
    vn3_3 = 0.5 * (v3_0 * t0_3 + v3_1 * t1_3 + v3_2 * t2_3 + v3_3 * t3_3 + v3_4 * t4_3 + v3_5 * t5_3)
    vn3_4 = 0.5 * (v3_0 * t0_4 + v3_1 * t1_4 + v3_2 * t2_4 + v3_3 * t3_4 + v3_4 * t4_4 + v3_5 * t5_4)
    vn3_5 = 0.5 * (v3_0 * t0_5 + v3_1 * t1_5 + v3_2 * t2_5 + v3_3 * t3_5 + v3_4 * t4_5 + v3_5 * t5_5)
    vn4_0 = 0.5 * (v4_0 * t0_0 + v4_1 * t1_0 + v4_2 * t2_0 + v4_3 * t3_0 + v4_4 * t4_0 + v4_5 * t5_0)
    vn4_1 = 0.5 * (v4_0 * t0_1 + v4_1 * t1_1 + v4_2 * t2_1 + v4_3 * t3_1 + v4_4 * t4_1 + v4_5 * t5_1)
    vn4_2 = 0.5 * (v4_0 * t0_2 + v4_1 * t1_2 + v4_2 * t2_2 + v4_3 * t3_2 + v4_4 * t4_2 + v4_5 * t5_2)
    vn4_3 = 0.5 * (v4_0 * t0_3 + v4_1 * t1_3 + v4_2 * t2_3 + v4_3 * t3_3 + v4_4 * t4_3 + v4_5 * t5_3)
    vn4_4 = 0.5 * (v4_0 * t0_4 + v4_1 * t1_4 + v4_2 * t2_4 + v4_3 * t3_4 + v4_4 * t4_4 + v4_5 * t5_4)
    vn4_5 = 0.5 * (v4_0 * t0_5 + v4_1 * t1_5 + v4_2 * t2_5 + v4_3 * t3_5 + v4_4 * t4_5 + v4_5 * t5_5)
    vn5_0 = 0.5 * (v5_0 * t0_0 + v5_1 * t1_0 + v5_2 * t2_0 + v5_3 * t3_0 + v5_4 * t4_0 + v5_5 * t5_0)
    vn5_1 = 0.5 * (v5_0 * t0_1 + v5_1 * t1_1 + v5_2 * t2_1 + v5_3 * t3_1 + v5_4 * t4_1 + v5_5 * t5_1)
    vn5_2 = 0.5 * (v5_0 * t0_2 + v5_1 * t1_2 + v5_2 * t2_2 + v5_3 * t3_2 + v5_4 * t4_2 + v5_5 * t5_2)
    vn5_3 = 0.5 * (v5_0 * t0_3 + v5_1 * t1_3 + v5_2 * t2_3 + v5_3 * t3_3 + v5_4 * t4_3 + v5_5 * t5_3)
    vn5_4 = 0.5 * (v5_0 * t0_4 + v5_1 * t1_4 + v5_2 * t2_4 + v5_3 * t3_4 + v5_4 * t4_4 + v5_5 * t5_4)
    vn5_5 = 0.5 * (v5_0 * t0_5 + v5_1 * t1_5 + v5_2 * t2_5 + v5_3 * t3_5 + v5_4 * t4_5 + v5_5 * t5_5)
    v0_0 = vn0_0
    v0_1 = vn0_1
    v0_2 = vn0_2
    v0_3 = vn0_3
    v0_4 = vn0_4
    v0_5 = vn0_5
    v1_0 = vn1_0
    v1_1 = vn1_1
    v1_2 = vn1_2
    v1_3 = vn1_3
    v1_4 = vn1_4
    v1_5 = vn1_5
    v2_0 = vn2_0
    v2_1 = vn2_1
    v2_2 = vn2_2
    v2_3 = vn2_3
    v2_4 = vn2_4
    v2_5 = vn2_5
    v3_0 = vn3_0
    v3_1 = vn3_1
    v3_2 = vn3_2
    v3_3 = vn3_3
    v3_4 = vn3_4
    v3_5 = vn3_5
    v4_0 = vn4_0
    v4_1 = vn4_1
    v4_2 = vn4_2
    v4_3 = vn4_3
    v4_4 = vn4_4
    v4_5 = vn4_5
    v5_0 = vn5_0
    v5_1 = vn5_1
    v5_2 = vn5_2
    v5_3 = vn5_3
    v5_4 = vn5_4
    v5_5 = vn5_5
    # NS iter
    y0_0 = v0_0 * v0_0 + v1_0 * v1_0 + v2_0 * v2_0 + v3_0 * v3_0 + v4_0 * v4_0 + v5_0 * v5_0
    y0_1 = v0_0 * v0_1 + v1_0 * v1_1 + v2_0 * v2_1 + v3_0 * v3_1 + v4_0 * v4_1 + v5_0 * v5_1
    y0_2 = v0_0 * v0_2 + v1_0 * v1_2 + v2_0 * v2_2 + v3_0 * v3_2 + v4_0 * v4_2 + v5_0 * v5_2
    y0_3 = v0_0 * v0_3 + v1_0 * v1_3 + v2_0 * v2_3 + v3_0 * v3_3 + v4_0 * v4_3 + v5_0 * v5_3
    y0_4 = v0_0 * v0_4 + v1_0 * v1_4 + v2_0 * v2_4 + v3_0 * v3_4 + v4_0 * v4_4 + v5_0 * v5_4
    y0_5 = v0_0 * v0_5 + v1_0 * v1_5 + v2_0 * v2_5 + v3_0 * v3_5 + v4_0 * v4_5 + v5_0 * v5_5
    y1_0 = v0_1 * v0_0 + v1_1 * v1_0 + v2_1 * v2_0 + v3_1 * v3_0 + v4_1 * v4_0 + v5_1 * v5_0
    y1_1 = v0_1 * v0_1 + v1_1 * v1_1 + v2_1 * v2_1 + v3_1 * v3_1 + v4_1 * v4_1 + v5_1 * v5_1
    y1_2 = v0_1 * v0_2 + v1_1 * v1_2 + v2_1 * v2_2 + v3_1 * v3_2 + v4_1 * v4_2 + v5_1 * v5_2
    y1_3 = v0_1 * v0_3 + v1_1 * v1_3 + v2_1 * v2_3 + v3_1 * v3_3 + v4_1 * v4_3 + v5_1 * v5_3
    y1_4 = v0_1 * v0_4 + v1_1 * v1_4 + v2_1 * v2_4 + v3_1 * v3_4 + v4_1 * v4_4 + v5_1 * v5_4
    y1_5 = v0_1 * v0_5 + v1_1 * v1_5 + v2_1 * v2_5 + v3_1 * v3_5 + v4_1 * v4_5 + v5_1 * v5_5
    y2_0 = v0_2 * v0_0 + v1_2 * v1_0 + v2_2 * v2_0 + v3_2 * v3_0 + v4_2 * v4_0 + v5_2 * v5_0
    y2_1 = v0_2 * v0_1 + v1_2 * v1_1 + v2_2 * v2_1 + v3_2 * v3_1 + v4_2 * v4_1 + v5_2 * v5_1
    y2_2 = v0_2 * v0_2 + v1_2 * v1_2 + v2_2 * v2_2 + v3_2 * v3_2 + v4_2 * v4_2 + v5_2 * v5_2
    y2_3 = v0_2 * v0_3 + v1_2 * v1_3 + v2_2 * v2_3 + v3_2 * v3_3 + v4_2 * v4_3 + v5_2 * v5_3
    y2_4 = v0_2 * v0_4 + v1_2 * v1_4 + v2_2 * v2_4 + v3_2 * v3_4 + v4_2 * v4_4 + v5_2 * v5_4
    y2_5 = v0_2 * v0_5 + v1_2 * v1_5 + v2_2 * v2_5 + v3_2 * v3_5 + v4_2 * v4_5 + v5_2 * v5_5
    y3_0 = v0_3 * v0_0 + v1_3 * v1_0 + v2_3 * v2_0 + v3_3 * v3_0 + v4_3 * v4_0 + v5_3 * v5_0
    y3_1 = v0_3 * v0_1 + v1_3 * v1_1 + v2_3 * v2_1 + v3_3 * v3_1 + v4_3 * v4_1 + v5_3 * v5_1
    y3_2 = v0_3 * v0_2 + v1_3 * v1_2 + v2_3 * v2_2 + v3_3 * v3_2 + v4_3 * v4_2 + v5_3 * v5_2
    y3_3 = v0_3 * v0_3 + v1_3 * v1_3 + v2_3 * v2_3 + v3_3 * v3_3 + v4_3 * v4_3 + v5_3 * v5_3
    y3_4 = v0_3 * v0_4 + v1_3 * v1_4 + v2_3 * v2_4 + v3_3 * v3_4 + v4_3 * v4_4 + v5_3 * v5_4
    y3_5 = v0_3 * v0_5 + v1_3 * v1_5 + v2_3 * v2_5 + v3_3 * v3_5 + v4_3 * v4_5 + v5_3 * v5_5
    y4_0 = v0_4 * v0_0 + v1_4 * v1_0 + v2_4 * v2_0 + v3_4 * v3_0 + v4_4 * v4_0 + v5_4 * v5_0
    y4_1 = v0_4 * v0_1 + v1_4 * v1_1 + v2_4 * v2_1 + v3_4 * v3_1 + v4_4 * v4_1 + v5_4 * v5_1
    y4_2 = v0_4 * v0_2 + v1_4 * v1_2 + v2_4 * v2_2 + v3_4 * v3_2 + v4_4 * v4_2 + v5_4 * v5_2
    y4_3 = v0_4 * v0_3 + v1_4 * v1_3 + v2_4 * v2_3 + v3_4 * v3_3 + v4_4 * v4_3 + v5_4 * v5_3
    y4_4 = v0_4 * v0_4 + v1_4 * v1_4 + v2_4 * v2_4 + v3_4 * v3_4 + v4_4 * v4_4 + v5_4 * v5_4
    y4_5 = v0_4 * v0_5 + v1_4 * v1_5 + v2_4 * v2_5 + v3_4 * v3_5 + v4_4 * v4_5 + v5_4 * v5_5
    y5_0 = v0_5 * v0_0 + v1_5 * v1_0 + v2_5 * v2_0 + v3_5 * v3_0 + v4_5 * v4_0 + v5_5 * v5_0
    y5_1 = v0_5 * v0_1 + v1_5 * v1_1 + v2_5 * v2_1 + v3_5 * v3_1 + v4_5 * v4_1 + v5_5 * v5_1
    y5_2 = v0_5 * v0_2 + v1_5 * v1_2 + v2_5 * v2_2 + v3_5 * v3_2 + v4_5 * v4_2 + v5_5 * v5_2
    y5_3 = v0_5 * v0_3 + v1_5 * v1_3 + v2_5 * v2_3 + v3_5 * v3_3 + v4_5 * v4_3 + v5_5 * v5_3
    y5_4 = v0_5 * v0_4 + v1_5 * v1_4 + v2_5 * v2_4 + v3_5 * v3_4 + v4_5 * v4_4 + v5_5 * v5_4
    y5_5 = v0_5 * v0_5 + v1_5 * v1_5 + v2_5 * v2_5 + v3_5 * v3_5 + v4_5 * v4_5 + v5_5 * v5_5
    t0_0 = 3.0 - y0_0
    t0_1 = -y0_1
    t0_2 = -y0_2
    t0_3 = -y0_3
    t0_4 = -y0_4
    t0_5 = -y0_5
    t1_0 = -y1_0
    t1_1 = 3.0 - y1_1
    t1_2 = -y1_2
    t1_3 = -y1_3
    t1_4 = -y1_4
    t1_5 = -y1_5
    t2_0 = -y2_0
    t2_1 = -y2_1
    t2_2 = 3.0 - y2_2
    t2_3 = -y2_3
    t2_4 = -y2_4
    t2_5 = -y2_5
    t3_0 = -y3_0
    t3_1 = -y3_1
    t3_2 = -y3_2
    t3_3 = 3.0 - y3_3
    t3_4 = -y3_4
    t3_5 = -y3_5
    t4_0 = -y4_0
    t4_1 = -y4_1
    t4_2 = -y4_2
    t4_3 = -y4_3
    t4_4 = 3.0 - y4_4
    t4_5 = -y4_5
    t5_0 = -y5_0
    t5_1 = -y5_1
    t5_2 = -y5_2
    t5_3 = -y5_3
    t5_4 = -y5_4
    t5_5 = 3.0 - y5_5
    vn0_0 = 0.5 * (v0_0 * t0_0 + v0_1 * t1_0 + v0_2 * t2_0 + v0_3 * t3_0 + v0_4 * t4_0 + v0_5 * t5_0)
    vn0_1 = 0.5 * (v0_0 * t0_1 + v0_1 * t1_1 + v0_2 * t2_1 + v0_3 * t3_1 + v0_4 * t4_1 + v0_5 * t5_1)
    vn0_2 = 0.5 * (v0_0 * t0_2 + v0_1 * t1_2 + v0_2 * t2_2 + v0_3 * t3_2 + v0_4 * t4_2 + v0_5 * t5_2)
    vn0_3 = 0.5 * (v0_0 * t0_3 + v0_1 * t1_3 + v0_2 * t2_3 + v0_3 * t3_3 + v0_4 * t4_3 + v0_5 * t5_3)
    vn0_4 = 0.5 * (v0_0 * t0_4 + v0_1 * t1_4 + v0_2 * t2_4 + v0_3 * t3_4 + v0_4 * t4_4 + v0_5 * t5_4)
    vn0_5 = 0.5 * (v0_0 * t0_5 + v0_1 * t1_5 + v0_2 * t2_5 + v0_3 * t3_5 + v0_4 * t4_5 + v0_5 * t5_5)
    vn1_0 = 0.5 * (v1_0 * t0_0 + v1_1 * t1_0 + v1_2 * t2_0 + v1_3 * t3_0 + v1_4 * t4_0 + v1_5 * t5_0)
    vn1_1 = 0.5 * (v1_0 * t0_1 + v1_1 * t1_1 + v1_2 * t2_1 + v1_3 * t3_1 + v1_4 * t4_1 + v1_5 * t5_1)
    vn1_2 = 0.5 * (v1_0 * t0_2 + v1_1 * t1_2 + v1_2 * t2_2 + v1_3 * t3_2 + v1_4 * t4_2 + v1_5 * t5_2)
    vn1_3 = 0.5 * (v1_0 * t0_3 + v1_1 * t1_3 + v1_2 * t2_3 + v1_3 * t3_3 + v1_4 * t4_3 + v1_5 * t5_3)
    vn1_4 = 0.5 * (v1_0 * t0_4 + v1_1 * t1_4 + v1_2 * t2_4 + v1_3 * t3_4 + v1_4 * t4_4 + v1_5 * t5_4)
    vn1_5 = 0.5 * (v1_0 * t0_5 + v1_1 * t1_5 + v1_2 * t2_5 + v1_3 * t3_5 + v1_4 * t4_5 + v1_5 * t5_5)
    vn2_0 = 0.5 * (v2_0 * t0_0 + v2_1 * t1_0 + v2_2 * t2_0 + v2_3 * t3_0 + v2_4 * t4_0 + v2_5 * t5_0)
    vn2_1 = 0.5 * (v2_0 * t0_1 + v2_1 * t1_1 + v2_2 * t2_1 + v2_3 * t3_1 + v2_4 * t4_1 + v2_5 * t5_1)
    vn2_2 = 0.5 * (v2_0 * t0_2 + v2_1 * t1_2 + v2_2 * t2_2 + v2_3 * t3_2 + v2_4 * t4_2 + v2_5 * t5_2)
    vn2_3 = 0.5 * (v2_0 * t0_3 + v2_1 * t1_3 + v2_2 * t2_3 + v2_3 * t3_3 + v2_4 * t4_3 + v2_5 * t5_3)
    vn2_4 = 0.5 * (v2_0 * t0_4 + v2_1 * t1_4 + v2_2 * t2_4 + v2_3 * t3_4 + v2_4 * t4_4 + v2_5 * t5_4)
    vn2_5 = 0.5 * (v2_0 * t0_5 + v2_1 * t1_5 + v2_2 * t2_5 + v2_3 * t3_5 + v2_4 * t4_5 + v2_5 * t5_5)
    vn3_0 = 0.5 * (v3_0 * t0_0 + v3_1 * t1_0 + v3_2 * t2_0 + v3_3 * t3_0 + v3_4 * t4_0 + v3_5 * t5_0)
    vn3_1 = 0.5 * (v3_0 * t0_1 + v3_1 * t1_1 + v3_2 * t2_1 + v3_3 * t3_1 + v3_4 * t4_1 + v3_5 * t5_1)
    vn3_2 = 0.5 * (v3_0 * t0_2 + v3_1 * t1_2 + v3_2 * t2_2 + v3_3 * t3_2 + v3_4 * t4_2 + v3_5 * t5_2)
    vn3_3 = 0.5 * (v3_0 * t0_3 + v3_1 * t1_3 + v3_2 * t2_3 + v3_3 * t3_3 + v3_4 * t4_3 + v3_5 * t5_3)
    vn3_4 = 0.5 * (v3_0 * t0_4 + v3_1 * t1_4 + v3_2 * t2_4 + v3_3 * t3_4 + v3_4 * t4_4 + v3_5 * t5_4)
    vn3_5 = 0.5 * (v3_0 * t0_5 + v3_1 * t1_5 + v3_2 * t2_5 + v3_3 * t3_5 + v3_4 * t4_5 + v3_5 * t5_5)
    vn4_0 = 0.5 * (v4_0 * t0_0 + v4_1 * t1_0 + v4_2 * t2_0 + v4_3 * t3_0 + v4_4 * t4_0 + v4_5 * t5_0)
    vn4_1 = 0.5 * (v4_0 * t0_1 + v4_1 * t1_1 + v4_2 * t2_1 + v4_3 * t3_1 + v4_4 * t4_1 + v4_5 * t5_1)
    vn4_2 = 0.5 * (v4_0 * t0_2 + v4_1 * t1_2 + v4_2 * t2_2 + v4_3 * t3_2 + v4_4 * t4_2 + v4_5 * t5_2)
    vn4_3 = 0.5 * (v4_0 * t0_3 + v4_1 * t1_3 + v4_2 * t2_3 + v4_3 * t3_3 + v4_4 * t4_3 + v4_5 * t5_3)
    vn4_4 = 0.5 * (v4_0 * t0_4 + v4_1 * t1_4 + v4_2 * t2_4 + v4_3 * t3_4 + v4_4 * t4_4 + v4_5 * t5_4)
    vn4_5 = 0.5 * (v4_0 * t0_5 + v4_1 * t1_5 + v4_2 * t2_5 + v4_3 * t3_5 + v4_4 * t4_5 + v4_5 * t5_5)
    vn5_0 = 0.5 * (v5_0 * t0_0 + v5_1 * t1_0 + v5_2 * t2_0 + v5_3 * t3_0 + v5_4 * t4_0 + v5_5 * t5_0)
    vn5_1 = 0.5 * (v5_0 * t0_1 + v5_1 * t1_1 + v5_2 * t2_1 + v5_3 * t3_1 + v5_4 * t4_1 + v5_5 * t5_1)
    vn5_2 = 0.5 * (v5_0 * t0_2 + v5_1 * t1_2 + v5_2 * t2_2 + v5_3 * t3_2 + v5_4 * t4_2 + v5_5 * t5_2)
    vn5_3 = 0.5 * (v5_0 * t0_3 + v5_1 * t1_3 + v5_2 * t2_3 + v5_3 * t3_3 + v5_4 * t4_3 + v5_5 * t5_3)
    vn5_4 = 0.5 * (v5_0 * t0_4 + v5_1 * t1_4 + v5_2 * t2_4 + v5_3 * t3_4 + v5_4 * t4_4 + v5_5 * t5_4)
    vn5_5 = 0.5 * (v5_0 * t0_5 + v5_1 * t1_5 + v5_2 * t2_5 + v5_3 * t3_5 + v5_4 * t4_5 + v5_5 * t5_5)
    v0_0 = vn0_0
    v0_1 = vn0_1
    v0_2 = vn0_2
    v0_3 = vn0_3
    v0_4 = vn0_4
    v0_5 = vn0_5
    v1_0 = vn1_0
    v1_1 = vn1_1
    v1_2 = vn1_2
    v1_3 = vn1_3
    v1_4 = vn1_4
    v1_5 = vn1_5
    v2_0 = vn2_0
    v2_1 = vn2_1
    v2_2 = vn2_2
    v2_3 = vn2_3
    v2_4 = vn2_4
    v2_5 = vn2_5
    v3_0 = vn3_0
    v3_1 = vn3_1
    v3_2 = vn3_2
    v3_3 = vn3_3
    v3_4 = vn3_4
    v3_5 = vn3_5
    v4_0 = vn4_0
    v4_1 = vn4_1
    v4_2 = vn4_2
    v4_3 = vn4_3
    v4_4 = vn4_4
    v4_5 = vn4_5
    v5_0 = vn5_0
    v5_1 = vn5_1
    v5_2 = vn5_2
    v5_3 = vn5_3
    v5_4 = vn5_4
    v5_5 = vn5_5
    # Phase 5: Rayleigh
    af0_0 = a0_0.to(tl.float32)
    af0_1 = a0_1.to(tl.float32)
    af0_2 = a0_2.to(tl.float32)
    af0_3 = a0_3.to(tl.float32)
    af0_4 = a0_4.to(tl.float32)
    af0_5 = a0_5.to(tl.float32)
    af1_0 = a1_0.to(tl.float32)
    af1_1 = a1_1.to(tl.float32)
    af1_2 = a1_2.to(tl.float32)
    af1_3 = a1_3.to(tl.float32)
    af1_4 = a1_4.to(tl.float32)
    af1_5 = a1_5.to(tl.float32)
    af2_0 = a2_0.to(tl.float32)
    af2_1 = a2_1.to(tl.float32)
    af2_2 = a2_2.to(tl.float32)
    af2_3 = a2_3.to(tl.float32)
    af2_4 = a2_4.to(tl.float32)
    af2_5 = a2_5.to(tl.float32)
    af3_0 = a3_0.to(tl.float32)
    af3_1 = a3_1.to(tl.float32)
    af3_2 = a3_2.to(tl.float32)
    af3_3 = a3_3.to(tl.float32)
    af3_4 = a3_4.to(tl.float32)
    af3_5 = a3_5.to(tl.float32)
    af4_0 = a4_0.to(tl.float32)
    af4_1 = a4_1.to(tl.float32)
    af4_2 = a4_2.to(tl.float32)
    af4_3 = a4_3.to(tl.float32)
    af4_4 = a4_4.to(tl.float32)
    af4_5 = a4_5.to(tl.float32)
    af5_0 = a5_0.to(tl.float32)
    af5_1 = a5_1.to(tl.float32)
    af5_2 = a5_2.to(tl.float32)
    af5_3 = a5_3.to(tl.float32)
    af5_4 = a5_4.to(tl.float32)
    af5_5 = a5_5.to(tl.float32)
    lam0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _av = af0_0 * v0_0 + af0_1 * v1_0 + af0_2 * v2_0 + af0_3 * v3_0 + af0_4 * v4_0 + af0_5 * v5_0
    lam0 = lam0 + v0_0 * _av
    _av = af1_0 * v0_0 + af1_1 * v1_0 + af1_2 * v2_0 + af1_3 * v3_0 + af1_4 * v4_0 + af1_5 * v5_0
    lam0 = lam0 + v1_0 * _av
    _av = af2_0 * v0_0 + af2_1 * v1_0 + af2_2 * v2_0 + af2_3 * v3_0 + af2_4 * v4_0 + af2_5 * v5_0
    lam0 = lam0 + v2_0 * _av
    _av = af3_0 * v0_0 + af3_1 * v1_0 + af3_2 * v2_0 + af3_3 * v3_0 + af3_4 * v4_0 + af3_5 * v5_0
    lam0 = lam0 + v3_0 * _av
    _av = af4_0 * v0_0 + af4_1 * v1_0 + af4_2 * v2_0 + af4_3 * v3_0 + af4_4 * v4_0 + af4_5 * v5_0
    lam0 = lam0 + v4_0 * _av
    _av = af5_0 * v0_0 + af5_1 * v1_0 + af5_2 * v2_0 + af5_3 * v3_0 + af5_4 * v4_0 + af5_5 * v5_0
    lam0 = lam0 + v5_0 * _av
    lam0 = lam0 * sc.to(tl.float32)
    lam1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _av = af0_0 * v0_1 + af0_1 * v1_1 + af0_2 * v2_1 + af0_3 * v3_1 + af0_4 * v4_1 + af0_5 * v5_1
    lam1 = lam1 + v0_1 * _av
    _av = af1_0 * v0_1 + af1_1 * v1_1 + af1_2 * v2_1 + af1_3 * v3_1 + af1_4 * v4_1 + af1_5 * v5_1
    lam1 = lam1 + v1_1 * _av
    _av = af2_0 * v0_1 + af2_1 * v1_1 + af2_2 * v2_1 + af2_3 * v3_1 + af2_4 * v4_1 + af2_5 * v5_1
    lam1 = lam1 + v2_1 * _av
    _av = af3_0 * v0_1 + af3_1 * v1_1 + af3_2 * v2_1 + af3_3 * v3_1 + af3_4 * v4_1 + af3_5 * v5_1
    lam1 = lam1 + v3_1 * _av
    _av = af4_0 * v0_1 + af4_1 * v1_1 + af4_2 * v2_1 + af4_3 * v3_1 + af4_4 * v4_1 + af4_5 * v5_1
    lam1 = lam1 + v4_1 * _av
    _av = af5_0 * v0_1 + af5_1 * v1_1 + af5_2 * v2_1 + af5_3 * v3_1 + af5_4 * v4_1 + af5_5 * v5_1
    lam1 = lam1 + v5_1 * _av
    lam1 = lam1 * sc.to(tl.float32)
    lam2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _av = af0_0 * v0_2 + af0_1 * v1_2 + af0_2 * v2_2 + af0_3 * v3_2 + af0_4 * v4_2 + af0_5 * v5_2
    lam2 = lam2 + v0_2 * _av
    _av = af1_0 * v0_2 + af1_1 * v1_2 + af1_2 * v2_2 + af1_3 * v3_2 + af1_4 * v4_2 + af1_5 * v5_2
    lam2 = lam2 + v1_2 * _av
    _av = af2_0 * v0_2 + af2_1 * v1_2 + af2_2 * v2_2 + af2_3 * v3_2 + af2_4 * v4_2 + af2_5 * v5_2
    lam2 = lam2 + v2_2 * _av
    _av = af3_0 * v0_2 + af3_1 * v1_2 + af3_2 * v2_2 + af3_3 * v3_2 + af3_4 * v4_2 + af3_5 * v5_2
    lam2 = lam2 + v3_2 * _av
    _av = af4_0 * v0_2 + af4_1 * v1_2 + af4_2 * v2_2 + af4_3 * v3_2 + af4_4 * v4_2 + af4_5 * v5_2
    lam2 = lam2 + v4_2 * _av
    _av = af5_0 * v0_2 + af5_1 * v1_2 + af5_2 * v2_2 + af5_3 * v3_2 + af5_4 * v4_2 + af5_5 * v5_2
    lam2 = lam2 + v5_2 * _av
    lam2 = lam2 * sc.to(tl.float32)
    lam3 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _av = af0_0 * v0_3 + af0_1 * v1_3 + af0_2 * v2_3 + af0_3 * v3_3 + af0_4 * v4_3 + af0_5 * v5_3
    lam3 = lam3 + v0_3 * _av
    _av = af1_0 * v0_3 + af1_1 * v1_3 + af1_2 * v2_3 + af1_3 * v3_3 + af1_4 * v4_3 + af1_5 * v5_3
    lam3 = lam3 + v1_3 * _av
    _av = af2_0 * v0_3 + af2_1 * v1_3 + af2_2 * v2_3 + af2_3 * v3_3 + af2_4 * v4_3 + af2_5 * v5_3
    lam3 = lam3 + v2_3 * _av
    _av = af3_0 * v0_3 + af3_1 * v1_3 + af3_2 * v2_3 + af3_3 * v3_3 + af3_4 * v4_3 + af3_5 * v5_3
    lam3 = lam3 + v3_3 * _av
    _av = af4_0 * v0_3 + af4_1 * v1_3 + af4_2 * v2_3 + af4_3 * v3_3 + af4_4 * v4_3 + af4_5 * v5_3
    lam3 = lam3 + v4_3 * _av
    _av = af5_0 * v0_3 + af5_1 * v1_3 + af5_2 * v2_3 + af5_3 * v3_3 + af5_4 * v4_3 + af5_5 * v5_3
    lam3 = lam3 + v5_3 * _av
    lam3 = lam3 * sc.to(tl.float32)
    lam4 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _av = af0_0 * v0_4 + af0_1 * v1_4 + af0_2 * v2_4 + af0_3 * v3_4 + af0_4 * v4_4 + af0_5 * v5_4
    lam4 = lam4 + v0_4 * _av
    _av = af1_0 * v0_4 + af1_1 * v1_4 + af1_2 * v2_4 + af1_3 * v3_4 + af1_4 * v4_4 + af1_5 * v5_4
    lam4 = lam4 + v1_4 * _av
    _av = af2_0 * v0_4 + af2_1 * v1_4 + af2_2 * v2_4 + af2_3 * v3_4 + af2_4 * v4_4 + af2_5 * v5_4
    lam4 = lam4 + v2_4 * _av
    _av = af3_0 * v0_4 + af3_1 * v1_4 + af3_2 * v2_4 + af3_3 * v3_4 + af3_4 * v4_4 + af3_5 * v5_4
    lam4 = lam4 + v3_4 * _av
    _av = af4_0 * v0_4 + af4_1 * v1_4 + af4_2 * v2_4 + af4_3 * v3_4 + af4_4 * v4_4 + af4_5 * v5_4
    lam4 = lam4 + v4_4 * _av
    _av = af5_0 * v0_4 + af5_1 * v1_4 + af5_2 * v2_4 + af5_3 * v3_4 + af5_4 * v4_4 + af5_5 * v5_4
    lam4 = lam4 + v5_4 * _av
    lam4 = lam4 * sc.to(tl.float32)
    lam5 = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _av = af0_0 * v0_5 + af0_1 * v1_5 + af0_2 * v2_5 + af0_3 * v3_5 + af0_4 * v4_5 + af0_5 * v5_5
    lam5 = lam5 + v0_5 * _av
    _av = af1_0 * v0_5 + af1_1 * v1_5 + af1_2 * v2_5 + af1_3 * v3_5 + af1_4 * v4_5 + af1_5 * v5_5
    lam5 = lam5 + v1_5 * _av
    _av = af2_0 * v0_5 + af2_1 * v1_5 + af2_2 * v2_5 + af2_3 * v3_5 + af2_4 * v4_5 + af2_5 * v5_5
    lam5 = lam5 + v2_5 * _av
    _av = af3_0 * v0_5 + af3_1 * v1_5 + af3_2 * v2_5 + af3_3 * v3_5 + af3_4 * v4_5 + af3_5 * v5_5
    lam5 = lam5 + v3_5 * _av
    _av = af4_0 * v0_5 + af4_1 * v1_5 + af4_2 * v2_5 + af4_3 * v3_5 + af4_4 * v4_5 + af4_5 * v5_5
    lam5 = lam5 + v4_5 * _av
    _av = af5_0 * v0_5 + af5_1 * v1_5 + af5_2 * v2_5 + af5_3 * v3_5 + af5_4 * v4_5 + af5_5 * v5_5
    lam5 = lam5 + v5_5 * _av
    lam5 = lam5 * sc.to(tl.float32)
    # Sort
    p0 = tl.zeros((BLOCK_B,), dtype=tl.int32) + 0
    p1 = tl.zeros((BLOCK_B,), dtype=tl.int32) + 1
    p2 = tl.zeros((BLOCK_B,), dtype=tl.int32) + 2
    p3 = tl.zeros((BLOCK_B,), dtype=tl.int32) + 3
    p4 = tl.zeros((BLOCK_B,), dtype=tl.int32) + 4
    p5 = tl.zeros((BLOCK_B,), dtype=tl.int32) + 5
    _sw = lam0 > lam1
    _ea, _eb = lam0, lam1
    lam0 = tl.where(_sw, _eb, _ea)
    lam1 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p0, p1
    p0 = tl.where(_sw, _pb, _pa)
    p1 = tl.where(_sw, _pa, _pb)
    _sw = lam1 > lam2
    _ea, _eb = lam1, lam2
    lam1 = tl.where(_sw, _eb, _ea)
    lam2 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p1, p2
    p1 = tl.where(_sw, _pb, _pa)
    p2 = tl.where(_sw, _pa, _pb)
    _sw = lam2 > lam3
    _ea, _eb = lam2, lam3
    lam2 = tl.where(_sw, _eb, _ea)
    lam3 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p2, p3
    p2 = tl.where(_sw, _pb, _pa)
    p3 = tl.where(_sw, _pa, _pb)
    _sw = lam3 > lam4
    _ea, _eb = lam3, lam4
    lam3 = tl.where(_sw, _eb, _ea)
    lam4 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p3, p4
    p3 = tl.where(_sw, _pb, _pa)
    p4 = tl.where(_sw, _pa, _pb)
    _sw = lam4 > lam5
    _ea, _eb = lam4, lam5
    lam4 = tl.where(_sw, _eb, _ea)
    lam5 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p4, p5
    p4 = tl.where(_sw, _pb, _pa)
    p5 = tl.where(_sw, _pa, _pb)
    _sw = lam0 > lam1
    _ea, _eb = lam0, lam1
    lam0 = tl.where(_sw, _eb, _ea)
    lam1 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p0, p1
    p0 = tl.where(_sw, _pb, _pa)
    p1 = tl.where(_sw, _pa, _pb)
    _sw = lam1 > lam2
    _ea, _eb = lam1, lam2
    lam1 = tl.where(_sw, _eb, _ea)
    lam2 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p1, p2
    p1 = tl.where(_sw, _pb, _pa)
    p2 = tl.where(_sw, _pa, _pb)
    _sw = lam2 > lam3
    _ea, _eb = lam2, lam3
    lam2 = tl.where(_sw, _eb, _ea)
    lam3 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p2, p3
    p2 = tl.where(_sw, _pb, _pa)
    p3 = tl.where(_sw, _pa, _pb)
    _sw = lam3 > lam4
    _ea, _eb = lam3, lam4
    lam3 = tl.where(_sw, _eb, _ea)
    lam4 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p3, p4
    p3 = tl.where(_sw, _pb, _pa)
    p4 = tl.where(_sw, _pa, _pb)
    _sw = lam4 > lam5
    _ea, _eb = lam4, lam5
    lam4 = tl.where(_sw, _eb, _ea)
    lam5 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p4, p5
    p4 = tl.where(_sw, _pb, _pa)
    p5 = tl.where(_sw, _pa, _pb)
    _sw = lam0 > lam1
    _ea, _eb = lam0, lam1
    lam0 = tl.where(_sw, _eb, _ea)
    lam1 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p0, p1
    p0 = tl.where(_sw, _pb, _pa)
    p1 = tl.where(_sw, _pa, _pb)
    _sw = lam1 > lam2
    _ea, _eb = lam1, lam2
    lam1 = tl.where(_sw, _eb, _ea)
    lam2 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p1, p2
    p1 = tl.where(_sw, _pb, _pa)
    p2 = tl.where(_sw, _pa, _pb)
    _sw = lam2 > lam3
    _ea, _eb = lam2, lam3
    lam2 = tl.where(_sw, _eb, _ea)
    lam3 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p2, p3
    p2 = tl.where(_sw, _pb, _pa)
    p3 = tl.where(_sw, _pa, _pb)
    _sw = lam3 > lam4
    _ea, _eb = lam3, lam4
    lam3 = tl.where(_sw, _eb, _ea)
    lam4 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p3, p4
    p3 = tl.where(_sw, _pb, _pa)
    p4 = tl.where(_sw, _pa, _pb)
    _sw = lam4 > lam5
    _ea, _eb = lam4, lam5
    lam4 = tl.where(_sw, _eb, _ea)
    lam5 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p4, p5
    p4 = tl.where(_sw, _pb, _pa)
    p5 = tl.where(_sw, _pa, _pb)
    _sw = lam0 > lam1
    _ea, _eb = lam0, lam1
    lam0 = tl.where(_sw, _eb, _ea)
    lam1 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p0, p1
    p0 = tl.where(_sw, _pb, _pa)
    p1 = tl.where(_sw, _pa, _pb)
    _sw = lam1 > lam2
    _ea, _eb = lam1, lam2
    lam1 = tl.where(_sw, _eb, _ea)
    lam2 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p1, p2
    p1 = tl.where(_sw, _pb, _pa)
    p2 = tl.where(_sw, _pa, _pb)
    _sw = lam2 > lam3
    _ea, _eb = lam2, lam3
    lam2 = tl.where(_sw, _eb, _ea)
    lam3 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p2, p3
    p2 = tl.where(_sw, _pb, _pa)
    p3 = tl.where(_sw, _pa, _pb)
    _sw = lam3 > lam4
    _ea, _eb = lam3, lam4
    lam3 = tl.where(_sw, _eb, _ea)
    lam4 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p3, p4
    p3 = tl.where(_sw, _pb, _pa)
    p4 = tl.where(_sw, _pa, _pb)
    _sw = lam4 > lam5
    _ea, _eb = lam4, lam5
    lam4 = tl.where(_sw, _eb, _ea)
    lam5 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p4, p5
    p4 = tl.where(_sw, _pb, _pa)
    p5 = tl.where(_sw, _pa, _pb)
    _sw = lam0 > lam1
    _ea, _eb = lam0, lam1
    lam0 = tl.where(_sw, _eb, _ea)
    lam1 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p0, p1
    p0 = tl.where(_sw, _pb, _pa)
    p1 = tl.where(_sw, _pa, _pb)
    _sw = lam1 > lam2
    _ea, _eb = lam1, lam2
    lam1 = tl.where(_sw, _eb, _ea)
    lam2 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p1, p2
    p1 = tl.where(_sw, _pb, _pa)
    p2 = tl.where(_sw, _pa, _pb)
    _sw = lam2 > lam3
    _ea, _eb = lam2, lam3
    lam2 = tl.where(_sw, _eb, _ea)
    lam3 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p2, p3
    p2 = tl.where(_sw, _pb, _pa)
    p3 = tl.where(_sw, _pa, _pb)
    _sw = lam3 > lam4
    _ea, _eb = lam3, lam4
    lam3 = tl.where(_sw, _eb, _ea)
    lam4 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p3, p4
    p3 = tl.where(_sw, _pb, _pa)
    p4 = tl.where(_sw, _pa, _pb)
    _sw = lam4 > lam5
    _ea, _eb = lam4, lam5
    lam4 = tl.where(_sw, _eb, _ea)
    lam5 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p4, p5
    p4 = tl.where(_sw, _pb, _pa)
    p5 = tl.where(_sw, _pa, _pb)
    _sw = lam0 > lam1
    _ea, _eb = lam0, lam1
    lam0 = tl.where(_sw, _eb, _ea)
    lam1 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p0, p1
    p0 = tl.where(_sw, _pb, _pa)
    p1 = tl.where(_sw, _pa, _pb)
    _sw = lam1 > lam2
    _ea, _eb = lam1, lam2
    lam1 = tl.where(_sw, _eb, _ea)
    lam2 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p1, p2
    p1 = tl.where(_sw, _pb, _pa)
    p2 = tl.where(_sw, _pa, _pb)
    _sw = lam2 > lam3
    _ea, _eb = lam2, lam3
    lam2 = tl.where(_sw, _eb, _ea)
    lam3 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p2, p3
    p2 = tl.where(_sw, _pb, _pa)
    p3 = tl.where(_sw, _pa, _pb)
    _sw = lam3 > lam4
    _ea, _eb = lam3, lam4
    lam3 = tl.where(_sw, _eb, _ea)
    lam4 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p3, p4
    p3 = tl.where(_sw, _pb, _pa)
    p4 = tl.where(_sw, _pa, _pb)
    _sw = lam4 > lam5
    _ea, _eb = lam4, lam5
    lam4 = tl.where(_sw, _eb, _ea)
    lam5 = tl.where(_sw, _ea, _eb)
    _pa, _pb = p4, p5
    p4 = tl.where(_sw, _pb, _pa)
    p5 = tl.where(_sw, _pa, _pb)
    # Store
    _is0 = (p0 == 0)
    _is1 = (p0 == 1)
    _is2 = (p0 == 2)
    _is3 = (p0 == 3)
    _is4 = (p0 == 4)
    _is5 = (p0 == 5)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v0_0, _sv)
    _sv = tl.where(_is1, v0_1, _sv)
    _sv = tl.where(_is2, v0_2, _sv)
    _sv = tl.where(_is3, v0_3, _sv)
    _sv = tl.where(_is4, v0_4, _sv)
    _sv = tl.where(_is5, v0_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 0, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v1_0, _sv)
    _sv = tl.where(_is1, v1_1, _sv)
    _sv = tl.where(_is2, v1_2, _sv)
    _sv = tl.where(_is3, v1_3, _sv)
    _sv = tl.where(_is4, v1_4, _sv)
    _sv = tl.where(_is5, v1_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 6, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v2_0, _sv)
    _sv = tl.where(_is1, v2_1, _sv)
    _sv = tl.where(_is2, v2_2, _sv)
    _sv = tl.where(_is3, v2_3, _sv)
    _sv = tl.where(_is4, v2_4, _sv)
    _sv = tl.where(_is5, v2_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 12, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v3_0, _sv)
    _sv = tl.where(_is1, v3_1, _sv)
    _sv = tl.where(_is2, v3_2, _sv)
    _sv = tl.where(_is3, v3_3, _sv)
    _sv = tl.where(_is4, v3_4, _sv)
    _sv = tl.where(_is5, v3_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 18, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v4_0, _sv)
    _sv = tl.where(_is1, v4_1, _sv)
    _sv = tl.where(_is2, v4_2, _sv)
    _sv = tl.where(_is3, v4_3, _sv)
    _sv = tl.where(_is4, v4_4, _sv)
    _sv = tl.where(_is5, v4_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 24, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v5_0, _sv)
    _sv = tl.where(_is1, v5_1, _sv)
    _sv = tl.where(_is2, v5_2, _sv)
    _sv = tl.where(_is3, v5_3, _sv)
    _sv = tl.where(_is4, v5_4, _sv)
    _sv = tl.where(_is5, v5_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 30, _sv, mask=mask)
    _is0 = (p1 == 0)
    _is1 = (p1 == 1)
    _is2 = (p1 == 2)
    _is3 = (p1 == 3)
    _is4 = (p1 == 4)
    _is5 = (p1 == 5)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v0_0, _sv)
    _sv = tl.where(_is1, v0_1, _sv)
    _sv = tl.where(_is2, v0_2, _sv)
    _sv = tl.where(_is3, v0_3, _sv)
    _sv = tl.where(_is4, v0_4, _sv)
    _sv = tl.where(_is5, v0_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 1, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v1_0, _sv)
    _sv = tl.where(_is1, v1_1, _sv)
    _sv = tl.where(_is2, v1_2, _sv)
    _sv = tl.where(_is3, v1_3, _sv)
    _sv = tl.where(_is4, v1_4, _sv)
    _sv = tl.where(_is5, v1_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 7, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v2_0, _sv)
    _sv = tl.where(_is1, v2_1, _sv)
    _sv = tl.where(_is2, v2_2, _sv)
    _sv = tl.where(_is3, v2_3, _sv)
    _sv = tl.where(_is4, v2_4, _sv)
    _sv = tl.where(_is5, v2_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 13, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v3_0, _sv)
    _sv = tl.where(_is1, v3_1, _sv)
    _sv = tl.where(_is2, v3_2, _sv)
    _sv = tl.where(_is3, v3_3, _sv)
    _sv = tl.where(_is4, v3_4, _sv)
    _sv = tl.where(_is5, v3_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 19, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v4_0, _sv)
    _sv = tl.where(_is1, v4_1, _sv)
    _sv = tl.where(_is2, v4_2, _sv)
    _sv = tl.where(_is3, v4_3, _sv)
    _sv = tl.where(_is4, v4_4, _sv)
    _sv = tl.where(_is5, v4_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 25, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v5_0, _sv)
    _sv = tl.where(_is1, v5_1, _sv)
    _sv = tl.where(_is2, v5_2, _sv)
    _sv = tl.where(_is3, v5_3, _sv)
    _sv = tl.where(_is4, v5_4, _sv)
    _sv = tl.where(_is5, v5_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 31, _sv, mask=mask)
    _is0 = (p2 == 0)
    _is1 = (p2 == 1)
    _is2 = (p2 == 2)
    _is3 = (p2 == 3)
    _is4 = (p2 == 4)
    _is5 = (p2 == 5)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v0_0, _sv)
    _sv = tl.where(_is1, v0_1, _sv)
    _sv = tl.where(_is2, v0_2, _sv)
    _sv = tl.where(_is3, v0_3, _sv)
    _sv = tl.where(_is4, v0_4, _sv)
    _sv = tl.where(_is5, v0_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 2, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v1_0, _sv)
    _sv = tl.where(_is1, v1_1, _sv)
    _sv = tl.where(_is2, v1_2, _sv)
    _sv = tl.where(_is3, v1_3, _sv)
    _sv = tl.where(_is4, v1_4, _sv)
    _sv = tl.where(_is5, v1_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 8, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v2_0, _sv)
    _sv = tl.where(_is1, v2_1, _sv)
    _sv = tl.where(_is2, v2_2, _sv)
    _sv = tl.where(_is3, v2_3, _sv)
    _sv = tl.where(_is4, v2_4, _sv)
    _sv = tl.where(_is5, v2_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 14, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v3_0, _sv)
    _sv = tl.where(_is1, v3_1, _sv)
    _sv = tl.where(_is2, v3_2, _sv)
    _sv = tl.where(_is3, v3_3, _sv)
    _sv = tl.where(_is4, v3_4, _sv)
    _sv = tl.where(_is5, v3_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 20, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v4_0, _sv)
    _sv = tl.where(_is1, v4_1, _sv)
    _sv = tl.where(_is2, v4_2, _sv)
    _sv = tl.where(_is3, v4_3, _sv)
    _sv = tl.where(_is4, v4_4, _sv)
    _sv = tl.where(_is5, v4_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 26, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v5_0, _sv)
    _sv = tl.where(_is1, v5_1, _sv)
    _sv = tl.where(_is2, v5_2, _sv)
    _sv = tl.where(_is3, v5_3, _sv)
    _sv = tl.where(_is4, v5_4, _sv)
    _sv = tl.where(_is5, v5_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 32, _sv, mask=mask)
    _is0 = (p3 == 0)
    _is1 = (p3 == 1)
    _is2 = (p3 == 2)
    _is3 = (p3 == 3)
    _is4 = (p3 == 4)
    _is5 = (p3 == 5)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v0_0, _sv)
    _sv = tl.where(_is1, v0_1, _sv)
    _sv = tl.where(_is2, v0_2, _sv)
    _sv = tl.where(_is3, v0_3, _sv)
    _sv = tl.where(_is4, v0_4, _sv)
    _sv = tl.where(_is5, v0_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 3, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v1_0, _sv)
    _sv = tl.where(_is1, v1_1, _sv)
    _sv = tl.where(_is2, v1_2, _sv)
    _sv = tl.where(_is3, v1_3, _sv)
    _sv = tl.where(_is4, v1_4, _sv)
    _sv = tl.where(_is5, v1_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 9, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v2_0, _sv)
    _sv = tl.where(_is1, v2_1, _sv)
    _sv = tl.where(_is2, v2_2, _sv)
    _sv = tl.where(_is3, v2_3, _sv)
    _sv = tl.where(_is4, v2_4, _sv)
    _sv = tl.where(_is5, v2_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 15, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v3_0, _sv)
    _sv = tl.where(_is1, v3_1, _sv)
    _sv = tl.where(_is2, v3_2, _sv)
    _sv = tl.where(_is3, v3_3, _sv)
    _sv = tl.where(_is4, v3_4, _sv)
    _sv = tl.where(_is5, v3_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 21, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v4_0, _sv)
    _sv = tl.where(_is1, v4_1, _sv)
    _sv = tl.where(_is2, v4_2, _sv)
    _sv = tl.where(_is3, v4_3, _sv)
    _sv = tl.where(_is4, v4_4, _sv)
    _sv = tl.where(_is5, v4_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 27, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v5_0, _sv)
    _sv = tl.where(_is1, v5_1, _sv)
    _sv = tl.where(_is2, v5_2, _sv)
    _sv = tl.where(_is3, v5_3, _sv)
    _sv = tl.where(_is4, v5_4, _sv)
    _sv = tl.where(_is5, v5_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 33, _sv, mask=mask)
    _is0 = (p4 == 0)
    _is1 = (p4 == 1)
    _is2 = (p4 == 2)
    _is3 = (p4 == 3)
    _is4 = (p4 == 4)
    _is5 = (p4 == 5)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v0_0, _sv)
    _sv = tl.where(_is1, v0_1, _sv)
    _sv = tl.where(_is2, v0_2, _sv)
    _sv = tl.where(_is3, v0_3, _sv)
    _sv = tl.where(_is4, v0_4, _sv)
    _sv = tl.where(_is5, v0_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 4, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v1_0, _sv)
    _sv = tl.where(_is1, v1_1, _sv)
    _sv = tl.where(_is2, v1_2, _sv)
    _sv = tl.where(_is3, v1_3, _sv)
    _sv = tl.where(_is4, v1_4, _sv)
    _sv = tl.where(_is5, v1_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 10, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v2_0, _sv)
    _sv = tl.where(_is1, v2_1, _sv)
    _sv = tl.where(_is2, v2_2, _sv)
    _sv = tl.where(_is3, v2_3, _sv)
    _sv = tl.where(_is4, v2_4, _sv)
    _sv = tl.where(_is5, v2_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 16, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v3_0, _sv)
    _sv = tl.where(_is1, v3_1, _sv)
    _sv = tl.where(_is2, v3_2, _sv)
    _sv = tl.where(_is3, v3_3, _sv)
    _sv = tl.where(_is4, v3_4, _sv)
    _sv = tl.where(_is5, v3_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 22, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v4_0, _sv)
    _sv = tl.where(_is1, v4_1, _sv)
    _sv = tl.where(_is2, v4_2, _sv)
    _sv = tl.where(_is3, v4_3, _sv)
    _sv = tl.where(_is4, v4_4, _sv)
    _sv = tl.where(_is5, v4_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 28, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v5_0, _sv)
    _sv = tl.where(_is1, v5_1, _sv)
    _sv = tl.where(_is2, v5_2, _sv)
    _sv = tl.where(_is3, v5_3, _sv)
    _sv = tl.where(_is4, v5_4, _sv)
    _sv = tl.where(_is5, v5_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 34, _sv, mask=mask)
    _is0 = (p5 == 0)
    _is1 = (p5 == 1)
    _is2 = (p5 == 2)
    _is3 = (p5 == 3)
    _is4 = (p5 == 4)
    _is5 = (p5 == 5)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v0_0, _sv)
    _sv = tl.where(_is1, v0_1, _sv)
    _sv = tl.where(_is2, v0_2, _sv)
    _sv = tl.where(_is3, v0_3, _sv)
    _sv = tl.where(_is4, v0_4, _sv)
    _sv = tl.where(_is5, v0_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 5, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v1_0, _sv)
    _sv = tl.where(_is1, v1_1, _sv)
    _sv = tl.where(_is2, v1_2, _sv)
    _sv = tl.where(_is3, v1_3, _sv)
    _sv = tl.where(_is4, v1_4, _sv)
    _sv = tl.where(_is5, v1_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 11, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v2_0, _sv)
    _sv = tl.where(_is1, v2_1, _sv)
    _sv = tl.where(_is2, v2_2, _sv)
    _sv = tl.where(_is3, v2_3, _sv)
    _sv = tl.where(_is4, v2_4, _sv)
    _sv = tl.where(_is5, v2_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 17, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v3_0, _sv)
    _sv = tl.where(_is1, v3_1, _sv)
    _sv = tl.where(_is2, v3_2, _sv)
    _sv = tl.where(_is3, v3_3, _sv)
    _sv = tl.where(_is4, v3_4, _sv)
    _sv = tl.where(_is5, v3_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 23, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v4_0, _sv)
    _sv = tl.where(_is1, v4_1, _sv)
    _sv = tl.where(_is2, v4_2, _sv)
    _sv = tl.where(_is3, v4_3, _sv)
    _sv = tl.where(_is4, v4_4, _sv)
    _sv = tl.where(_is5, v4_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 29, _sv, mask=mask)
    _sv = tl.zeros((BLOCK_B,), dtype=tl.float32)
    _sv = tl.where(_is0, v5_0, _sv)
    _sv = tl.where(_is1, v5_1, _sv)
    _sv = tl.where(_is2, v5_2, _sv)
    _sv = tl.where(_is3, v5_3, _sv)
    _sv = tl.where(_is4, v5_4, _sv)
    _sv = tl.where(_is5, v5_5, _sv)
    tl.store(evecs_ptr + bid * 36 + 35, _sv, mask=mask)
    tl.store(evals_ptr + bid * 6 + 0, lam0, mask=mask)
    tl.store(evals_ptr + bid * 6 + 1, lam1, mask=mask)
    tl.store(evals_ptr + bid * 6 + 2, lam2, mask=mask)
    tl.store(evals_ptr + bid * 6 + 3, lam3, mask=mask)
    tl.store(evals_ptr + bid * 6 + 4, lam4, mask=mask)
    tl.store(evals_ptr + bid * 6 + 5, lam5, mask=mask)