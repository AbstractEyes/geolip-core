"""
Conv + SVD observation encoder.

4-stage conv backbone with detached SVD observation taps.
Proven: 70.92% CIFAR-100 with 3.9M params.

See experiment 8.21 for the training run.

todo: migrate from experiment_8_21_svd_test.py
"""
