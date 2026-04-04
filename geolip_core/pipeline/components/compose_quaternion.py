"""
QuaternionCompose — four-arm Hamilton product composition.
Proven in GeoQuat head (rho=0.309, 76/84 wins).
"""

import torch
import torch.nn as nn
from geolip_core.pipeline.observer import TorchComponent


def quaternion_multiply_batched(q1, q2):
    """Hamilton product on (B, 4, D) tensors. Fully vectorized."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=1)


class QuaternionCompose(TorchComponent):
    """Four-arm Hamilton product composition. Proven in GeoQuat head."""
    def __init__(self, name, input_dim, quat_dim=64):
        super().__init__(name)
        self.quat_dim = quat_dim
        self.proj_w = nn.Linear(input_dim, quat_dim)
        self.proj_i = nn.Linear(input_dim, quat_dim)
        self.proj_j = nn.Linear(input_dim, quat_dim)
        self.proj_k = nn.Linear(input_dim, quat_dim)
        self.rotation = nn.Parameter(torch.randn(1, 4, quat_dim) * 0.1)

    @property
    def output_dim(self):
        return self.quat_dim * 4

    def forward(self, arm_w, arm_i, arm_j, arm_k):
        shape = arm_w.shape[:-1]
        D = arm_w.shape[-1]
        flat = arm_w.dim() > 2
        if flat:
            arm_w = arm_w.reshape(-1, D); arm_i = arm_i.reshape(-1, D)
            arm_j = arm_j.reshape(-1, D); arm_k = arm_k.reshape(-1, D)
        q = torch.stack([self.proj_w(arm_w), self.proj_i(arm_i),
                         self.proj_j(arm_j), self.proj_k(arm_k)], dim=1)
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        r = self.rotation.expand(q.shape[0], -1, -1)
        r = r / (r.norm(dim=1, keepdim=True) + 1e-8)
        composed = quaternion_multiply_batched(r, q)
        composed = composed.reshape(q.shape[0], -1)
        if flat:
            composed = composed.reshape(*shape, -1)
        return composed
