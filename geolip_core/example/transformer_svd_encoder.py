"""
DeiT-Small + SVD observation encoder.

12-layer transformer with SVD taps at layers 3, 6, 9, 12.
Backbone-agnostic validation: +3.2 points over baseline.
See experiment 8.23 for the training run.

todo: migrate from experiment_8_23_deit_svd.py
"""
