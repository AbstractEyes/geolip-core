"""
CurriculumTrainer -- orchestrate episodic crystallization training.

Stage loop:
    Stage 0: Train transformer (CE only, no constellation losses) for N epochs
             -> Freeze model
             -> Crystallize constellation from stable embedding space

    Stage K: Inject constellation from Stage K-1
             -> Train with full observer losses for N epochs
             -> Freeze model
             -> Re-crystallize constellation
             -> Procrustes-align to Stage K-1's constellation
             -> Inherit stable anchors, reassign dead ones

Usage:
    trainer = CurriculumTrainer(model, config)
    history = trainer.train_curriculum(loader, stages=[
        {'epochs': 50, 'losses': ['ce']},
        {'epochs': 50, 'losses': 'all'},
    ])
"""

import torch
import torch.nn.functional as F

from geolip_core.core.align.crystallize import CrystallizationEngine
from geolip_core.core.align.stage_align import StageAligner


class CurriculumTrainer:
    """Orchestrate episodic crystallization training.

    The trainer does NOT own the training loop -- it accepts a user-provided
    `train_fn(model, loader, epochs, loss_config)` for flexibility.

    Args:
        model: GeometricTransformer or any model with constellation access
        config: dict with keys:
            manifold_dim (int): embedding dimension
            n_anchors (int): constellation anchor count
            n_classes (int): number of target classes
            project_fn (callable): model, inputs -> (B, manifold_dim) embeddings
            get_constellation_fn (callable): model -> Constellation module
            set_constellation_fn (callable): model, Constellation -> None
            train_fn (callable): model, loader, epochs, loss_config -> metrics
            rank (int, optional): Procrustes rank, default 24
            whiten (bool, optional): Procrustes whitening, default True
            stable_threshold (float, optional): default 0.1
            min_util (int, optional): minimum utilization, default 5
            max_samples (int, optional): cap for embedding collection, default 50000
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.engine = CrystallizationEngine(
            manifold_dim=config['manifold_dim'],
            n_anchors=config['n_anchors'],
            n_classes=config['n_classes'],
        )
        self.aligner = StageAligner(
            manifold_dim=config['manifold_dim'],
            rank=config.get('rank', 24),
            whiten=config.get('whiten', True),
            stable_threshold=config.get('stable_threshold', 0.1),
            min_util=config.get('min_util', 5),
        )

        self.project_fn = config['project_fn']
        self.get_constellation_fn = config['get_constellation_fn']
        self.set_constellation_fn = config['set_constellation_fn']
        self.train_fn = config['train_fn']
        self.max_samples = config.get('max_samples', 50000)

        self.history = []
        self.prev_constellation = None

    def inject_constellation(self, constellation):
        """Replace the model's constellation with a crystallized one.

        Uses the user-provided set_constellation_fn to handle model-specific
        details (shared vs per-layer constellations, etc.).
        """
        self.set_constellation_fn(self.model, constellation)

    def freeze_model(self):
        """Freeze all model parameters and switch to eval mode."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_model(self):
        """Unfreeze all model parameters and switch to train mode."""
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def crystallize(self, loader, device=None):
        """Crystallize constellation from current model state.

        Args:
            loader: data loader
            device: target device for constellation

        Returns:
            constellation: Constellation with crystallized anchors
            crystal_report: dict with crystallization diagnostics
        """
        self.freeze_model()
        constellation, report = self.engine.crystallize(
            self.model, loader, self.project_fn,
            device=device, max_samples=self.max_samples)
        self.unfreeze_model()
        return constellation, report

    def run_stage(self, stage_num, stage_config, loader, device=None):
        """Run one stage of curriculum training.

        Args:
            stage_num: stage index (0 = first stage, CE-only)
            stage_config: dict with 'epochs' and 'losses' keys
            loader: data loader
            device: target device

        Returns:
            report: dict with training metrics + crystallization diagnostics
        """
        report = {'stage': stage_num}

        # Inject constellation from previous stage (if available)
        if stage_num > 0 and self.prev_constellation is not None:
            self.inject_constellation(self.prev_constellation)
            report['injected_constellation'] = True

        # Train
        epochs = stage_config.get('epochs', 50)
        loss_config = stage_config.get('losses', ['ce'] if stage_num == 0 else 'all')
        train_metrics = self.train_fn(
            self.model, loader, epochs, loss_config)
        report['train_metrics'] = train_metrics

        # Crystallize
        constellation, crystal_report = self.crystallize(loader, device=device)
        report['crystallization'] = crystal_report

        # Align to previous stage if available
        if stage_num > 0 and self.prev_constellation is not None:
            # Collect embeddings for alignment
            self.freeze_model()
            embeddings, labels = self.engine.collect_embeddings(
                self.model, loader, self.project_fn,
                max_samples=self.max_samples)
            self.unfreeze_model()

            aligned_anchors, align_report = self.aligner.align(
                self.prev_constellation, embeddings, labels)

            # Reassign dead anchors
            if align_report['n_dead'] > 0:
                centroids, classes = self.engine.compute_centroids(embeddings, labels)
                confusion = self.engine.confusion_matrix(
                    embeddings, labels, centroids, classes)
                aligned_anchors = self.aligner.reassign_dead(
                    aligned_anchors, align_report['dead_mask'],
                    centroids, confusion)
                align_report['dead_reassigned'] = True

            # Structural drift report
            old_anchors = F.normalize(
                self.prev_constellation.anchors.data, dim=-1)
            drift_report = self.aligner.structural_report(
                old_anchors, aligned_anchors)
            align_report['drift'] = drift_report

            # Merge aligned anchors: keep stable from alignment, rest from crystallization
            stable = align_report['stable_mask']
            crystal_anchors = F.normalize(
                constellation.anchors.data, dim=-1)

            # Composite: stable anchors from alignment, drifting+dead from crystallization
            merged = crystal_anchors.clone()
            merged[stable] = aligned_anchors[stable]
            with torch.no_grad():
                constellation.anchors.data.copy_(F.normalize(merged, dim=-1))

            report['alignment'] = align_report

        # Store for next stage
        self.prev_constellation = constellation
        self.history.append(report)

        return report

    def train_curriculum(self, loader, stages, device=None):
        """Run full curriculum training.

        Args:
            loader: data loader
            stages: list of stage configs, e.g.:
                [
                    {'epochs': 50, 'losses': ['ce']},       # Stage 0: CE only
                    {'epochs': 50, 'losses': 'all'},         # Stage 1: full observer
                    {'epochs': 30, 'losses': 'all'},         # Stage 2: refinement
                ]
            device: target device

        Returns:
            history: list of stage reports
        """
        for i, stage_config in enumerate(stages):
            report = self.run_stage(i, stage_config, loader, device=device)

            # Log summary
            crystal = report.get('crystallization', {})
            cv = crystal.get('cv', 'N/A')
            n_stable = report.get('alignment', {}).get('n_stable', 'N/A')
            n_dead = report.get('alignment', {}).get('n_dead', 'N/A')
            print(f"  Stage {i}: CV={cv}, stable={n_stable}, dead={n_dead}")

        return self.history
