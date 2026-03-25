# @title Analysis — SVD Test Model Geometric Structure
#
# Probes the trained model_svd_test for:
#   1. Embedding geometry: CV, cosine similarity, effective dimension
#   2. Per-class structure: inter/intra class cosine, class separation
#   3. SVD feature analysis: S spectrum, Vh structure per depth
#   4. Feature attribution: how much does SVD vs conv contribute?
#   5. kNN accuracy at different k values

import numpy as np
from collections import defaultdict

import torch


@torch.no_grad()
def analyze_svd_model(model, val_loader, device, n_max=5000):
    """Comprehensive geometric analysis of the trained SVD test model."""
    model.eval()
    model = model.to(device)

    # ── Collect all features ──
    all_svd_feats = []     # per-depth SVD features
    all_conv_feats = []    # pooled conv features
    all_combined = []      # full classifier input
    all_logits = []
    all_labels = []
    all_S = [[], [], [], []]  # singular values per depth
    all_Vh = [[], [], [], []]  # rotation matrices per depth

    n_collected = 0
    for images, labels in val_loader:
        if n_collected >= n_max:
            break
        images, labels = images.to(device), labels.to(device)
        B = images.shape[0]

        # Run through stages manually to collect intermediates
        h = images
        svd_feats_batch = []
        for i, (stage, pool, proj) in enumerate(zip(model.stages, model.pools, model.to_svd)):
            h = stage(h)
            h_svd = proj(h)
            H, W = h_svd.shape[2], h_svd.shape[3]
            h_flat = h_svd.permute(0, 2, 3, 1).reshape(B, H * W, model.svd_rank)
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    _, S, Vh = gram_eigh_svd(h_flat.float())
                    S = S.clamp(min=1e-6)
            all_S[i].append(S.cpu())
            all_Vh[i].append(Vh.cpu())
            svd_feats_batch.append(model._extract_svd_features(S, Vh))
            h = pool(h)

        conv_feat = model.final_pool(h).flatten(1)
        combined = torch.cat(svd_feats_batch + [conv_feat], dim=-1)
        logits = model.classifier(combined)

        all_svd_feats.append(torch.cat(svd_feats_batch, dim=-1).cpu())
        all_conv_feats.append(conv_feat.cpu())
        all_combined.append(combined.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        n_collected += B

    svd_feats = torch.cat(all_svd_feats)[:n_max]
    conv_feats = torch.cat(all_conv_feats)[:n_max]
    combined = torch.cat(all_combined)[:n_max]
    logits = torch.cat(all_logits)[:n_max]
    labels = torch.cat(all_labels)[:n_max]
    S_all = [torch.cat(s)[:n_max] for s in all_S]
    Vh_all = [torch.cat(v)[:n_max] for v in all_Vh]

    acc = (logits.argmax(-1) == labels).float().mean().item() * 100
    n = svd_feats.shape[0]

    print(f"\n{'='*80}")
    print(f"  SVD MODEL GEOMETRIC ANALYSIS — {n} samples")
    print(f"  Val accuracy: {acc:.1f}%")
    print(f"{'='*80}")

    # ════════════════════════════════════════════════════════════════════
    # 1. EMBEDDING GEOMETRY
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  1. EMBEDDING GEOMETRY")
    print(f"{'─'*60}")

    for name, feats in [("SVD (264-d)", svd_feats), ("Conv (384-d)", conv_feats),
                         ("Combined (648-d)", combined)]:
        feats_n = F.normalize(feats.float(), dim=-1).to(device)

        # CV
        cv = cv_metric(feats_n, n_samples=200)

        # Cosine similarity distribution
        sub = feats_n[:min(2000, n)]
        sim = sub @ sub.T
        mask = ~torch.eye(sub.shape[0], dtype=torch.bool, device=device)
        cos_mean = sim[mask].mean().item()
        cos_std = sim[mask].std().item()

        # Effective dimension via SVD
        centered = feats.float()[:2000] - feats.float()[:2000].mean(0)
        sv = torch.linalg.svdvals(centered.to(device))
        sv_norm = sv / sv.sum()
        eff_dim = (sv.sum() ** 2 / (sv ** 2).sum()).item()
        top5_energy = sv_norm[:5].sum().item()
        top10_energy = sv_norm[:10].sum().item()
        top20_energy = sv_norm[:20].sum().item()

        print(f"\n  {name}:")
        print(f"    CV:            {cv:.4f}")
        print(f"    Cosine sim:    {cos_mean:.4f} ± {cos_std:.4f}")
        print(f"    Eff dimension: {eff_dim:.1f} / {feats.shape[1]}")
        print(f"    Energy top-5:  {top5_energy*100:.1f}%  top-10: {top10_energy*100:.1f}%  top-20: {top20_energy*100:.1f}%")

    # ════════════════════════════════════════════════════════════════════
    # 2. PER-CLASS STRUCTURE
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  2. PER-CLASS STRUCTURE (combined features)")
    print(f"{'─'*60}")

    combined_n = F.normalize(combined.float(), dim=-1).to(device)
    classes = labels.unique().sort().values

    # Class centroids
    centroids = []
    for c in classes:
        mask_c = labels == c
        if mask_c.sum() > 0:
            centroids.append(F.normalize(combined_n[mask_c].mean(0, keepdim=True), dim=-1))
    centroids = torch.cat(centroids)  # (n_classes, D)

    # Inter-class cosine (centroid-to-centroid)
    inter_sim = centroids @ centroids.T
    inter_mask = ~torch.eye(centroids.shape[0], dtype=torch.bool, device=device)
    inter_mean = inter_sim[inter_mask].mean().item()
    inter_std = inter_sim[inter_mask].std().item()
    inter_max = inter_sim[inter_mask].max().item()

    # Intra-class cosine (samples to their centroid)
    intra_cos = []
    for i, c in enumerate(classes):
        mask_c = labels == c
        if mask_c.sum() > 1:
            sims = (combined_n[mask_c] @ centroids[i]).mean().item()
            intra_cos.append(sims)
    intra_mean = np.mean(intra_cos)
    intra_std = np.std(intra_cos)

    # Class separation ratio
    sep_ratio = intra_mean / (inter_mean + 1e-8)

    print(f"  Inter-class cosine: {inter_mean:.4f} ± {inter_std:.4f}  (max: {inter_max:.4f})")
    print(f"  Intra-class cosine: {intra_mean:.4f} ± {intra_std:.4f}")
    print(f"  Separation ratio:   {sep_ratio:.2f}x  (higher = better separated)")

    # ════════════════════════════════════════════════════════════════════
    # 3. SVD FEATURE ANALYSIS PER DEPTH
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  3. SVD SPECTRUM PER DEPTH")
    print(f"{'─'*60}")

    spatial_names = ["32×32", "16×16", "8×8", "4×4"]
    for i in range(4):
        S = S_all[i]  # (n, k)
        Vh = Vh_all[i]  # (n, k, k)
        k = S.shape[1]

        # Singular value statistics
        s_mean = S.mean(0)
        s_std = S.std(0)

        # Energy concentration
        s_norm = S / S.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        top1_energy = s_norm[:, 0].mean().item()
        top5_energy = s_norm[:, :5].sum(-1).mean().item()
        top10_energy = s_norm[:, :min(10, k)].sum(-1).mean().item()

        # Spectral entropy
        s_ent = -(s_norm * (s_norm.clamp(min=1e-8)).log()).sum(-1).mean().item()
        max_ent = math.log(k)

        # Vh structure: how diagonal is the rotation?
        vh_diag = Vh.diagonal(dim1=-2, dim2=-1)  # (n, k)
        diag_energy = vh_diag.pow(2).sum(-1).mean().item()
        total_energy = Vh.pow(2).sum((-2, -1)).mean().item()
        diag_ratio = diag_energy / (total_energy + 1e-8)

        # Condition number proxy: ratio of largest to smallest S
        cond = (S[:, 0] / S[:, -1].clamp(min=1e-8)).mean().item()

        print(f"\n  Depth {i} ({spatial_names[i]}, rank={k}):")
        print(f"    S mean:     [{', '.join(f'{v:.2f}' for v in s_mean[:6].tolist())}{'...' if k > 6 else ''}]")
        print(f"    Energy:     top-1={top1_energy*100:.1f}%  top-5={top5_energy*100:.1f}%  top-10={top10_energy*100:.1f}%")
        print(f"    Entropy:    {s_ent:.3f} / {max_ent:.3f}  ({s_ent/max_ent*100:.0f}% of max)")
        print(f"    Vh diag:    {diag_ratio*100:.1f}% energy on diagonal")
        print(f"    Cond ratio: {cond:.1f}x")

    # ════════════════════════════════════════════════════════════════════
    # 4. FEATURE ATTRIBUTION — SVD vs Conv
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  4. FEATURE ATTRIBUTION — SVD vs Conv")
    print(f"{'─'*60}")

    # Test classifier with zeroed-out SVD or conv features
    model_device = next(model.parameters()).device

    # Full features
    full_logits = model.classifier(combined.to(model_device))
    full_acc = (full_logits.argmax(-1) == labels.to(model_device)).float().mean().item() * 100

    # Zero SVD features (first 264 dims)
    no_svd = combined.clone()
    no_svd[:, :264] = 0.0
    no_svd_logits = model.classifier(no_svd.to(model_device))
    no_svd_acc = (no_svd_logits.argmax(-1) == labels.to(model_device)).float().mean().item() * 100

    # Zero conv features (last 384 dims)
    no_conv = combined.clone()
    no_conv[:, 264:] = 0.0
    no_conv_logits = model.classifier(no_conv.to(model_device))
    no_conv_acc = (no_conv_logits.argmax(-1) == labels.to(model_device)).float().mean().item() * 100

    # Per-depth SVD ablation
    depth_accs = []
    for d in range(4):
        ablated = combined.clone()
        start = d * 66
        ablated[:, start:start+66] = 0.0
        abl_logits = model.classifier(ablated.to(model_device))
        abl_acc = (abl_logits.argmax(-1) == labels.to(model_device)).float().mean().item() * 100
        depth_accs.append(abl_acc)

    print(f"  Full features:      {full_acc:.1f}%")
    print(f"  Zero SVD (264-d):   {no_svd_acc:.1f}%  (drop: {full_acc - no_svd_acc:+.1f})")
    print(f"  Zero conv (384-d):  {no_conv_acc:.1f}%  (drop: {full_acc - no_conv_acc:+.1f})")
    print(f"  SVD contribution:   {full_acc - no_svd_acc:.1f} points")
    print(f"  Conv contribution:  {full_acc - no_conv_acc:.1f} points")
    print(f"\n  Per-depth SVD ablation (zero one depth at a time):")
    for d in range(4):
        drop = full_acc - depth_accs[d]
        print(f"    Zero depth {d}: {depth_accs[d]:.1f}%  (drop: {drop:+.1f})")

    # ════════════════════════════════════════════════════════════════════
    # 5. kNN ACCURACY
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  5. kNN ACCURACY")
    print(f"{'─'*60}")

    for name, feats in [("SVD only", svd_feats), ("Conv only", conv_feats),
                         ("Combined", combined)]:
        feats_n = F.normalize(feats.float(), dim=-1).to(device)
        sub = feats_n[:min(5000, n)]
        sub_labels = labels[:min(5000, n)].to(device)

        for k_val in [1, 5, 10]:
            knn = knn_accuracy(sub, sub_labels, k=k_val) * 100
            print(f"  {name:>12} kNN-{k_val}: {knn:.1f}%")
        print()

    # ════════════════════════════════════════════════════════════════════
    # 6. GEOMETRIC CONSTANTS CHECK
    # ════════════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"  6. GEOMETRIC CONSTANTS")
    print(f"{'─'*60}")

    combined_n = F.normalize(combined.float(), dim=-1).to(device)

    # Multi-scale CV
    cv_scales = cv_multi_scale(combined_n[:2000], scales=(3, 4, 5, 6, 7, 8))
    print(f"  CV multi-scale: {cv_scales}")

    # Check for 0.20-0.23 universal attractor band
    cv5 = cv_scales.get(5, None)
    if cv5 is not None:
        in_band = 0.20 <= cv5 <= 0.23
        print(f"  CV pentachoron: {cv5:.4f} — {'IN BAND [0.20-0.23]' if in_band else 'outside band'}")

    # Norm distribution
    norms = combined.float().norm(dim=-1)
    print(f"  Embedding norms: {norms.mean():.4f} ± {norms.std():.4f}")

    print(f"\n{'='*80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*80}")


# ── Run ──────────────────────────────────────────────────────────────────────

analyze_svd_model(model_svd_test, val_loader, device)