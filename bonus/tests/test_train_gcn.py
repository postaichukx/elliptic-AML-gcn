from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

import train_gcn


class TrainGcnConfigTests(unittest.TestCase):
    def test_hyperparameters_are_global_variables(self) -> None:
        self.assertEqual(train_gcn.DATA_DIR, "data/elliptic")
        self.assertEqual(train_gcn.ITERATIONS, 200)
        self.assertEqual(train_gcn.HIDDEN_CHANNELS, 128)
        self.assertEqual(train_gcn.DROPOUT, 0.5)
        self.assertEqual(train_gcn.LEARNING_RATE, 0.01)
        self.assertEqual(train_gcn.WEIGHT_DECAY, 5e-4)
        self.assertEqual(train_gcn.VALIDATION_SIZE, 0.15)
        self.assertEqual(train_gcn.SEED, 42)
        self.assertEqual(train_gcn.EVAL_EVERY, 5)
        self.assertEqual(train_gcn.PATIENCE, 8)
        self.assertEqual(train_gcn.MIN_DELTA, 0.0)
        self.assertEqual(train_gcn.DECISION_THRESHOLD, 0.65)
        self.assertTrue(train_gcn.MAKE_UNDIRECTED)

    def test_parser_api_is_removed(self) -> None:
        self.assertFalse(hasattr(train_gcn, "build_" + "parser"))
        self.assertFalse(hasattr(train_gcn, "parse_" + "args"))


class TrainGcnPlotTests(unittest.TestCase):
    def test_plot_training_shows_loss_f1_and_confusion_matrix(self) -> None:
        history = [
            {"iteration": 1, "train_loss": 0.9, "val_loss": 1.0, "val_f1_illicit": 0.2},
            {"iteration": 2, "train_loss": 0.7, "val_loss": 0.8, "val_f1_illicit": 0.4},
        ]
        test_metrics = {"confusion_matrix": [[10, 2], [3, 5]]}

        figure = train_gcn.plot_training(history, test_metrics, show=False)
        try:
            axes = figure.axes
            self.assertEqual(len(axes), 4)
            self.assertEqual(axes[0].get_xlabel(), "Iteration")
            self.assertEqual(axes[0].get_ylabel(), "Loss")
            self.assertEqual(axes[1].get_xlabel(), "Iteration")
            self.assertEqual(axes[1].get_ylabel(), "F1-score")
            self.assertIn("Test confusion matrix", axes[2].get_title())
            self.assertEqual(axes[3].get_ylabel(), "<colorbar>")
        finally:
            plt.close(figure)

    def test_confusion_matrix_displays_total_accuracy_and_cell_percentages(self) -> None:
        history = [
            {"iteration": 1, "train_loss": 0.9, "val_loss": 1.0, "val_f1_illicit": 0.2},
        ]
        test_metrics = {"confusion_matrix": [[10, 2], [3, 5]]}

        figure = train_gcn.plot_training(history, test_metrics, show=False)
        try:
            confusion_axis = figure.axes[2]
            cell_labels = [text.get_text() for text in confusion_axis.texts]

            self.assertIn("Correct: 75.00% (15/20)", confusion_axis.get_title())
            self.assertIn("10\n50.00%", cell_labels)
            self.assertIn("5\n25.00%", cell_labels)
        finally:
            plt.close(figure)


class TrainGcnEarlyStoppingTests(unittest.TestCase):
    def test_early_stopping_stops_after_patience_without_improvement(self) -> None:
        stopper = train_gcn.EarlyStopping(patience=2, min_delta=0.01)

        self.assertTrue(stopper.update(1, 0.50))
        self.assertTrue(stopper.update(2, 0.515))
        self.assertFalse(stopper.update(3, 0.520))
        self.assertTrue(stopper.update(4, 0.526))
        self.assertEqual(stopper.best_iteration, 4)


class TrainGcnSummaryTests(unittest.TestCase):
    def test_format_summary_uses_iteration_wording(self) -> None:
        stats = {
            "num_nodes": 5,
            "num_edges": 4,
            "num_features": 3,
            "all_labels": {"licit": 2, "illicit": 1, "unknown": 2},
            "train_labels": {"licit": 1, "illicit": 1, "unknown": 0},
            "val_labels": {"licit": 1, "illicit": 0, "unknown": 0},
            "test_labels": {"licit": 1, "illicit": 0, "unknown": 0},
        }
        test_metrics = {
            "threshold": 0.5,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision_illicit": 1.0,
            "recall_illicit": 1.0,
            "f1_illicit": 1.0,
            "roc_auc": 1.0,
            "average_precision": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
        }
        threshold_info = {"f1_illicit": 1.0}

        summary = train_gcn.format_summary(stats, test_metrics, 7, threshold_info)

        self.assertIn("Best validation iteration: 7", summary)
        self.assertNotIn("ep" + "och", summary.lower())


if __name__ == "__main__":
    unittest.main()
