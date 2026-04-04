"""
training -- Training orchestration for episodic crystallization.

Curriculum training separates the constellation from gradient-based training:
    Stage 0: Train transformer (CE only) -> freeze -> crystallize constellation
    Stage K: Train with crystallized constellation (full observer losses)
             -> freeze -> Procrustes-align new space to old constellation
             -> inherit stable anchors, prune dead ones
"""

from .curriculum import CurriculumTrainer
