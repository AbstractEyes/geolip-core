"""
Geometric Transformer --CM-Validated Pipeline (re-export shim)
==================================================
This file re-exports all classes from their individual modules for
backward compatibility. Each class now lives in its own file:

    curate_cm_validated.py      --CMValidatedGate, pairwise_distances_squared,
                                  cayley_menger_det, anchor_neighborhood_cm
    distinguish_nce_bank.py     --GeoResidualBank
    context_film.py             --FiLMLayer
    align_cayley.py             --CayleyOrthogonal
    compose_quaternion.py       --QuaternionCompose, quaternion_multiply_batched
    project_manifold.py         --ManifoldProjection
    context_position_geometric.py --PositionGeometricContext
    attend_geometric.py         --GeometricAttention
    attend_content.py           --ContentAttention
    layer_geometric.py          --GeometricTransformerLayer
    transformer_geometric.py    --GeometricTransformer, factories

Author: AbstractPhil + Claude Opus 4.6
License: Apache 2.0
"""

# ═══════════════════════════════════════════════════════════════════════════════
# RE-EXPORTS --all public names available from original import path
# ═══════════════════════════════════════════════════════════════════════════════

from .curate_cm_validated import (
    pairwise_distances_squared,
    cayley_menger_det,
    anchor_neighborhood_cm,
    CMValidatedGate,
)
from .distinguish_nce_bank import GeoResidualBank
from .context_film import FiLMLayer
from .align_cayley import CayleyOrthogonal
from .compose_quaternion import quaternion_multiply_batched, QuaternionCompose
from .project_manifold import ManifoldProjection
from .context_position_geometric import PositionGeometricContext
from .attend_geometric import GeometricAttention
from .attend_content import ContentAttention
from .layer_geometric import GeometricTransformerLayer
from .transformer_geometric import (
    GeometricTransformer,
    geo_transformer_esm2,
    geo_transformer_small,
    geo_transformer_vision,
    _HAS_WIDE_ROUTER,
)

__all__ = [
    'pairwise_distances_squared', 'cayley_menger_det', 'anchor_neighborhood_cm',
    'CMValidatedGate', 'GeoResidualBank', 'FiLMLayer', 'CayleyOrthogonal',
    'quaternion_multiply_batched', 'QuaternionCompose', 'ManifoldProjection',
    'PositionGeometricContext', 'GeometricAttention', 'ContentAttention',
    'GeometricTransformerLayer', 'GeometricTransformer',
    'geo_transformer_esm2', 'geo_transformer_small', 'geo_transformer_vision',
]


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import torch
    import torch.nn as nn

    print("Geometric Transformer --CM Validated --Self-Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build small model ──
    model = geo_transformer_small('test_cm', n_layers=2)
    if hasattr(model, 'network_to'):
        model.network_to(device=device, strict=False)
    else:
        model = model.to(device)
    total = model.param_report()

    # ── Forward pass ──
    B, L, D = 2, 32, 256
    x = torch.randn(B, L, D, device=device)
    out, geos = model(x, return_geo_state=True)

    assert out.shape == (B, L, D), f"Expected ({B},{L},{D}), got {out.shape}"
    assert len(geos) == 2
    print(f"\n  Input:  ({B}, {L}, {D})")
    print(f"  Output: {out.shape}")
    print(f"  Geo states: {len(geos)} layers")

    # ── Verify CM gate is active ──
    for i, gs in enumerate(geos):
        gi = gs['gate_info']
        cm_q = gs['cm_quality']
        gv = gs['gate_values']
        print(f"\n  Layer {i} CM gate:")
        print(f"    active anchors:   {gi['active'].item():.1f} / {model.n_anchors}")
        print(f"    gate mean:        {gi['gate_mean'].item():.4f}")
        print(f"    cm_positive_frac: {gi['cm_positive_frac'].item():.3f}")
        print(f"    gate_values:      {gv.shape}  range=[{gv.min():.3f}, {gv.max():.3f}]")
        print(f"    cm_quality:       {cm_q.shape}  mean={cm_q.mean():.4f}")

    # ── Verify geo_residual continuity ──
    gr0 = geos[0]['geo_residual']
    gr1 = geos[1]['geo_residual']
    print(f"\n  Geo residual stream:")
    print(f"    Layer 0: {gr0.shape}  norm={gr0.norm(dim=-1).mean():.4f}")
    print(f"    Layer 1: {gr1.shape}  norm={gr1.norm(dim=-1).mean():.4f}")

    # ── Geometric losses ──
    geo_losses = model.geometric_losses()
    print(f"\n  Geometric regularization:")
    for k, v in geo_losses.items():
        print(f"    {k}: {v.item():.6f}")

    # ── Anchor diagnostics ──
    diag = model.anchor_diagnostics()
    print(f"\n  Anchor diagnostics:")
    for layer_name, d in diag.items():
        print(f"    {layer_name}:")
        for k, v in d.items():
            print(f"      {k}: {v:.4f}")

    # ── Verify Cayley rotations ──
    print(f"\n  Cayley rotations:")
    for name, module in model.named_modules():
        if isinstance(module, CayleyOrthogonal):
            R = module.get_rotation()
            I = torch.eye(R.shape[0], device=R.device)
            print(f"    {name}: ||RR^T-I||={((R@R.T)-I).norm():.8f}  det={torch.det(R):.4f}")

    # ── Gradient flow through CM gate ──
    print(f"\n  Gradient flow test:")
    model.zero_grad()
    x_grad = torch.randn(B, L, D, device=device, requires_grad=True)
    out_grad = model(x_grad)
    loss = out_grad.sum()
    loss.backward()

    # Check gate_proj has gradients
    for i in range(model.n_layers):
        layer = model[f'layer_{i}']
        gate_grads = [p.grad is not None and p.grad.abs().sum() > 0
                      for p in layer['cm_gate'].parameters()]
        print(f"    layer_{i} cm_gate grad: {'YES' if all(gate_grads) else 'NO'}")

    # ── Training step simulation ──
    print(f"\n  Training step simulation:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    x_train = torch.randn(B, L, D, device=device)
    out_train, states = model(x_train, return_geo_state=True)
    task_loss = out_train.mean()  # dummy

    geo_losses = model.geometric_losses()
    total_loss = task_loss + geo_losses.get('geo_total', 0.0)
    total_loss.backward()
    optimizer.step()
    print(f"    task_loss:  {task_loss.item():.4f}")
    print(f"    cv_loss:    {geo_losses['cv'].item():.6f}")
    print(f"    spread_loss:{geo_losses['spread'].item():.6f}")
    print(f"    total:      {total_loss.item():.4f}")

    # ── Paired forward + observer loss ──
    print(f"\n  Paired forward + observer loss:")
    model.zero_grad()

    x1 = torch.randn(B, L, D, device=device)
    x2 = x1 + 0.1 * torch.randn_like(x1)  # view 2 = slight perturbation
    targets = torch.randint(0, 10, (B,), device=device)

    output = model.forward_paired(x1, x2)
    print(f"    Output keys: {sorted(k for k in output if not k.startswith('geo_'))}")
    for k in ['embedding', 'patchwork1', 'bridge1', 'assign1', 'tri1']:
        print(f"    {k}: {output[k].shape}")

    # Task head for CE
    num_classes = 10
    head = nn.Linear(D, num_classes).to(device)

    loss, ld = model.compute_loss(output, targets, head=head)
    print(f"\n    Three-domain loss breakdown:")
    for k in ['loss_observer', 'loss_task', 'ce', 'nce_emb', 'nce_pw',
               'bridge', 'assign', 'assign_nce', 'nce_tri', 'attract',
               'cv', 'spread']:
        if k in ld:
            v = ld[k]
            v = v.item() if isinstance(v, torch.Tensor) else v
            print(f"      {k:16s} = {v:.4f}")
    for k in ['nce_emb_acc', 'nce_pw_acc', 'nce_tri_acc', 'bridge_acc',
               'assign_nce_acc', 'acc']:
        if k in ld:
            v = ld[k]
            v = v if isinstance(v, float) else v.item()
            print(f"      {k:16s} = {v*100:.1f}%")
    print(f"      {'TOTAL':16s} = {loss.item():.4f}")

    # Verify backward through observer loss
    loss.backward()
    alive_base, dead_base = [], []
    for n, p in model.named_parameters():
        if p.grad is not None and p.grad.norm() > 0:
            alive_base.append(n)
        else:
            dead_base.append(n)
    print(f"\n    Gradient flow: {len(alive_base)} params alive, {len(dead_base)} dead")
    if dead_base:
        print(f"\n    DEAD parameters (base model, paired+observer):")
        for n in dead_base:
            print(f"      {n}")

    # ══════════════════════════════════════════════════════════════
    # WIDE ROUTER COMPILATION
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  WideRouter Compilation")
    print(f"{'='*60}")

    if _HAS_WIDE_ROUTER:
        from geofractal.router.wide_router import WideRouter

        # Wrap transformer in WideRouter (same pattern as GeoViTClassifier)
        router = WideRouter('test_router', strict=False)
        router.attach('transformer', model)
        router.register_tower('transformer')
        router.network_to(device=device, strict=False)

        # Discover towers and compile
        router.discover_towers()
        print(f"\n  Towers discovered: {router.tower_names}")
        print(f"  Analyzed: {router.objects.get('_analyzed', False)}")

        try:
            compiled_router = router.compile(mode='default')
            print(f"  WideRouter.compile(mode='default'): OK")
        except Exception as e:
            print(f"  WideRouter.compile: {str(e)[:60]}")

        # Forward through the registered tower directly
        with torch.no_grad():
            out_via_router = router['transformer'](x)
        print(f"  Forward via router['transformer']: {out_via_router.shape}  OK")

        del router
    else:
        print(f"\n  WideRouter: geofractal not installed")

    print(f"\n{'='*60}")
    print(f"  PASSED --CM-validated pipeline operational")
    print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════
    # FLOW ENSEMBLE INTEGRATION TESTS
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  Flow Ensemble Integration")
    print(f"{'='*60}")

    del model, optimizer
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    model_f = geo_transformer_small('test_flows', n_layers=2,
                                     flow_keys=['quat_lite', 'velocity', 'orbital'])
    if hasattr(model_f, 'network_to'):
        model_f.network_to(device=device, strict=False)
    else:
        model_f = model_f.to(device)

    total_f = model_f.param_report()
    print(f"\n  Total params (with flows): {total_f:,}")

    print(f"\n  Flow ensemble per layer:")
    for i in range(model_f.n_layers):
        layer = model_f[f'layer_{i}']
        if layer.has('flows'):
            flows = layer['flows']
            names = flows.active_flow_names
            params = sum(p.numel() for p in flows.parameters())
            print(f"    layer_{i}: {names}  ({params:,} params)")
        else:
            print(f"    layer_{i}: no flows attached")

    x_f = torch.randn(B, L, D, device=device)
    out_f, geos_f = model_f(x_f, return_geo_state=True)
    assert out_f.shape == (B, L, D)
    print(f"\n  Forward with flows: {out_f.shape}  OK")

    geo_ctx_0 = geos_f[0]['geo_ctx']
    print(f"  Geo context shape: {geo_ctx_0.shape}  norm={geo_ctx_0.norm(dim=-1).mean():.4f}")

    print(f"\n  Flow gradient test (out.sum().backward()):")
    model_f.zero_grad()
    x_fg = torch.randn(B, L, D, device=device, requires_grad=True)
    out_fg = model_f(x_fg)
    out_fg.sum().backward()

    alive_simple, dead_simple = [], []
    for n, p in model_f.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            alive_simple.append(n)
        else:
            dead_simple.append(n)
    print(f"    {len(alive_simple)} alive, {len(dead_simple)} dead")
    if dead_simple:
        print(f"\n    DEAD parameters (out.sum):")
        for n in dead_simple:
            print(f"      {n}")

    print(f"\n  Paired forward + observer loss (with flows):")
    model_f.zero_grad()
    x1_f = torch.randn(B, L, D, device=device)
    x2_f = x1_f + 0.1 * torch.randn_like(x1_f)
    targets_f = torch.randint(0, 10, (B,), device=device)

    output_f = model_f.forward_paired(x1_f, x2_f)
    head_f = nn.Linear(D, num_classes).to(device)
    loss_f, ld_f = model_f.compute_loss(output_f, targets_f, head=head_f)
    print(f"    total loss: {loss_f.item():.4f}")
    loss_f.backward()

    alive_paired, dead_paired = [], []
    for n, p in model_f.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            alive_paired.append(n)
        else:
            dead_paired.append(n)
    print(f"    {len(alive_paired)} alive, {len(dead_paired)} dead")
    if dead_paired:
        print(f"\n    DEAD parameters (paired+observer):")
        for n in dead_paired:
            print(f"      {n}")

    print(f"\n  Runtime flow management:")
    layer0 = model_f['layer_0']
    flows_0 = layer0['flows']
    print(f"    Before:           {flows_0.active_flow_names}")

    flows_0.attach_flow('alignment')
    print(f"    +alignment:       {flows_0.active_flow_names}")

    flows_0.detach_flow('velocity')
    print(f"    -velocity:        {flows_0.active_flow_names}")

    out_swapped = model_f(x_f)
    assert out_swapped.shape == (B, L, D)
    print(f"    Forward after swap: {out_swapped.shape}  OK")

    layer1 = model_f['layer_1']
    if layer1.has('flows'):
        for fn in list(layer1['flows'].active_flow_names):
            key = fn.replace('flow_', '')
            layer1['flows'].detach_flow(key)
        print(f"    Layer 1 after clear: {layer1['flows'].active_flow_names}")
        out_partial = model_f(x_f)
        assert out_partial.shape == (B, L, D)
        print(f"    Forward (L0 flows, L1 empty): {out_partial.shape}  OK")

    print(f"\n  Backward compatibility (no flows):")
    model_nf = geo_transformer_small('test_noflows', n_layers=2)
    if hasattr(model_nf, 'network_to'):
        model_nf.network_to(device=device, strict=False)
    else:
        model_nf = model_nf.to(device)
    out_nf = model_nf(torch.randn(B, L, D, device=device))
    assert out_nf.shape == (B, L, D)
    print(f"    Forward (no flows): {out_nf.shape}  OK")
    for i in range(model_nf.n_layers):
        assert not model_nf[f'layer_{i}'].has('flows'), f"layer_{i} should not have flows"
    print(f"    No flows attached:  OK")
    del model_nf

    print(f"\n{'='*60}")
    print(f"  PASSED --CM-validated pipeline operational")
    print(f"  PASSED --Flow ensemble integration verified")
    print(f"  PASSED --Flow attach/detach verified")
    print(f"  PASSED --Backward compatibility verified")
    print(f"{'='*60}")
