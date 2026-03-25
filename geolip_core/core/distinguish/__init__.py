"""
distinguish — Task-specific output and training signal.

Loss functions and task heads. These are the irreducible minimum:
CE for the task, diagnostics for health monitoring.
"""

from .losses import (
    cv_loss, cv_metric, cv_multi_scale, cayley_menger_vol2,
    nce_loss, ce_loss, ce_loss_paired,
    bridge_loss, bridge_loss_paired,
    assign_bce_loss, assign_nce_loss,
    attraction_loss, spread_loss, knn_accuracy,
    three_domain_loss, observer_loss,
)
