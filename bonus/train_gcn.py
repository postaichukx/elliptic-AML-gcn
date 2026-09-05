from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


CLASS_NAMES = {0: "licit", 1: "illicit", 2: "unknown"}

DATA_DIR = "data/elliptic"
ITERATIONS = 200
HIDDEN_CHANNELS = 128
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
VALIDATION_SIZE = 0.15
SEED = 42
EVAL_EVERY = 5
DEVICE = "cpu"
NUM_THREADS = 0
MAKE_UNDIRECTED = True
DOWNLOAD_ONLY = False
TUNE_THRESHOLD = False
DECISION_THRESHOLD = 0.65
PATIENCE = 8
MIN_DELTA = 0.0
DISABLE_EARLY_STOPPING = False
SHOW_PLOT = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


class GCN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, 2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x)


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float = -float("inf")
    best_iteration: int = 0
    checks_without_improvement: int = 0

    def update(self, iteration: int, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_iteration = iteration
            self.checks_without_improvement = 0
            return True

        self.checks_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.checks_without_improvement >= self.patience


def load_elliptic_dataset(data_dir: str, make_undirected: bool) -> Any:
    dataset = EllipticBitcoinDataset(root=data_dir)
    data = dataset[0]
    data.x = data.x.float()

    if make_undirected:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    return data


def make_train_val_masks(
    data: Any, val_size: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labeled_train_idx = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    labeled_train_y = data.y[data.train_mask].cpu().numpy()

    train_idx, val_idx = train_test_split(
        labeled_train_idx,
        test_size=val_size,
        random_state=seed,
        stratify=labeled_train_y,
    )

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[torch.as_tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.as_tensor(val_idx, dtype=torch.long)] = True

    return train_mask, val_mask, data.test_mask.clone()


def normalize_features(data: Any, train_mask: torch.Tensor) -> Any:
    train_x = data.x[train_mask]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    data.x = (data.x - mean) / std

    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        raise ValueError("Feature normalization produced NaN or infinite values.")

    return data


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    return counts.sum() / (2.0 * counts.clamp_min(1.0))


def count_labels(y: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, int]:
    labels = y if mask is None else y[mask]
    counts = torch.bincount(labels.cpu(), minlength=3)
    return {CLASS_NAMES[i]: int(counts[i].item()) for i in range(3)}


def dataset_stats(data: Any, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> dict[str, Any]:
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.num_features),
        "all_labels": count_labels(data.y),
        "train_labels": count_labels(data.y, train_mask),
        "val_labels": count_labels(data.y, val_mask),
        "test_labels": count_labels(data.y, test_mask),
    }


@torch.no_grad()
def evaluate(
    model: GCN,
    data: Any,
    mask: torch.Tensor,
    weights: torch.Tensor,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[mask], data.y[mask], weight=weights)
    probabilities = logits[mask].softmax(dim=-1)[:, 1].cpu().numpy()
    predictions = (probabilities >= threshold).astype(np.int64)
    truth = data.y[mask].cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision_licit": float(precision[0]),
        "recall_licit": float(recall[0]),
        "f1_licit": float(f1[0]),
        "support_licit": int(support[0]),
        "precision_illicit": float(precision[1]),
        "recall_illicit": float(recall[1]),
        "f1_illicit": float(f1[1]),
        "support_illicit": int(support[1]),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
    }

    if len(np.unique(truth)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(truth, probabilities))
        metrics["average_precision"] = float(average_precision_score(truth, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


@torch.no_grad()
def find_best_threshold(model: GCN, data: Any, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index)
    probabilities = logits[mask].softmax(dim=-1)[:, 1].cpu().numpy()
    truth = data.y[mask].cpu().numpy()

    best = {
        "threshold": DECISION_THRESHOLD,
        "precision_illicit": 0.0,
        "recall_illicit": 0.0,
        "f1_illicit": -1.0,
    }

    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )
        candidate = {
            "threshold": float(threshold),
            "precision_illicit": float(precision[1]),
            "recall_illicit": float(recall[1]),
            "f1_illicit": float(f1[1]),
        }
        if candidate["f1_illicit"] > best["f1_illicit"]:
            best = candidate

    return best


def plot_training(
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    show: bool = True,
) -> plt.Figure:
    iterations = [row["iteration"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_rows = [row for row in history if "val_loss" in row]

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(iterations, train_loss, label="train loss")
    if val_rows:
        axes[0].plot(
            [row["iteration"] for row in val_rows],
            [row["val_loss"] for row in val_rows],
            marker="o",
            label="validation loss",
        )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("GCN training loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if val_rows:
        axes[1].plot(
            [row["iteration"] for row in val_rows],
            [row["val_f1_illicit"] for row in val_rows],
            marker="o",
            label="validation illicit F1",
        )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("F1-score")
    axes[1].set_title("Validation F1 for illicit class")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    matrix = np.array(test_metrics["confusion_matrix"])
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    total_accuracy = 100.0 * correct / total if total else 0.0
    image = axes[2].imshow(matrix, cmap="Blues")
    axes[2].set_title(f"Test confusion matrix\nCorrect: {total_accuracy:.2f}% ({correct}/{total})")
    axes[2].set_xticks([0, 1], ["licit", "illicit"])
    axes[2].set_yticks([0, 1], ["licit", "illicit"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            cell_percent = 100.0 * matrix[row, column] / total if total else 0.0
            axes[2].text(
                column,
                row,
                f"{matrix[row, column]}\n{cell_percent:.2f}%",
                ha="center",
                va="center",
                color="black",
            )
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("<colorbar>")

    figure.tight_layout()
    if show:
        plt.show()
    return figure


def format_summary(
    stats: dict[str, Any],
    test_metrics: dict[str, Any],
    best_iteration: int,
    threshold_info: dict[str, float],
) -> str:
    lines = [
        "Elliptic GCN run summary",
        "",
        f"Nodes: {stats['num_nodes']}",
        f"Edges: {stats['num_edges']}",
        f"Features: {stats['num_features']}",
        f"Labels: {stats['all_labels']}",
        f"Train labels: {stats['train_labels']}",
        f"Validation labels: {stats['val_labels']}",
        f"Test labels: {stats['test_labels']}",
        "",
        f"Best validation iteration: {best_iteration}",
        f"Decision threshold: {test_metrics['threshold']:.2f}",
        f"Validation illicit F1 at threshold: {threshold_info['f1_illicit']:.4f}",
        f"Test accuracy: {test_metrics['accuracy']:.4f}",
        f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"Test illicit precision: {test_metrics['precision_illicit']:.4f}",
        f"Test illicit recall: {test_metrics['recall_illicit']:.4f}",
        f"Test illicit F1: {test_metrics['f1_illicit']:.4f}",
        f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] is not None else "Test ROC-AUC: n/a",
        f"Test average precision: {test_metrics['average_precision']:.4f}"
        if test_metrics["average_precision"] is not None
        else "Test average precision: n/a",
        f"Confusion matrix [[TN, FP], [FN, TP]]: {test_metrics['confusion_matrix']}",
    ]
    return "\n".join(lines)


def train_model() -> tuple[dict[str, Any], dict[str, Any], int, dict[str, float], list[dict[str, Any]]]:
    set_seed(SEED)
    if NUM_THREADS > 0:
        torch.set_num_threads(NUM_THREADS)

    data = load_elliptic_dataset(DATA_DIR, MAKE_UNDIRECTED)
    train_mask, val_mask, test_mask = make_train_val_masks(data, VALIDATION_SIZE, SEED)
    stats = dataset_stats(data, train_mask, val_mask, test_mask)

    print(json.dumps(stats, indent=2))
    if DOWNLOAD_ONLY:
        return stats, {}, 0, {"threshold": DECISION_THRESHOLD, "f1_illicit": 0.0}, []

    data = normalize_features(data, train_mask)
    device = choose_device(DEVICE)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    weights = class_weights(data.y[train_mask]).to(device)
    model = GCN(data.num_features, HIDDEN_CHANNELS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_iteration = 0
    best_state = deepcopy(model.state_dict())
    early_stopping = EarlyStopping(PATIENCE, MIN_DELTA)
    history: list[dict[str, Any]] = []

    for iteration in range(1, ITERATIONS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        row: dict[str, Any] = {"iteration": iteration, "train_loss": float(loss.item())}

        should_evaluate = iteration == 1 or iteration == ITERATIONS or iteration % EVAL_EVERY == 0
        if should_evaluate:
            val_metrics = evaluate(model, data, val_mask, weights)
            row["val_loss"] = val_metrics["loss"]
            row["val_f1_illicit"] = val_metrics["f1_illicit"]
            row["val_recall_illicit"] = val_metrics["recall_illicit"]

            improved = early_stopping.update(iteration, val_metrics["f1_illicit"])
            if improved:
                best_iteration = iteration
                best_state = deepcopy(model.state_dict())

            print(
                f"Iteration {iteration:03d} | train loss {loss.item():.4f} | "
                f"val illicit F1 {val_metrics['f1_illicit']:.4f} | "
                f"val illicit recall {val_metrics['recall_illicit']:.4f}"
            )
            if not DISABLE_EARLY_STOPPING and early_stopping.should_stop:
                history.append(row)
                print(
                    f"Early stopping at iteration {iteration}: "
                    f"best validation illicit F1 {early_stopping.best_score:.4f} "
                    f"at iteration {early_stopping.best_iteration}."
                )
                break
        else:
            print(f"Iteration {iteration:03d} | train loss {loss.item():.4f}")

        history.append(row)

    model.load_state_dict(best_state)

    if TUNE_THRESHOLD:
        threshold_info = find_best_threshold(model, data, val_mask)
    else:
        val_metrics = evaluate(model, data, val_mask, weights, threshold=DECISION_THRESHOLD)
        threshold_info = {
            "threshold": DECISION_THRESHOLD,
            "precision_illicit": val_metrics["precision_illicit"],
            "recall_illicit": val_metrics["recall_illicit"],
            "f1_illicit": val_metrics["f1_illicit"],
        }

    test_metrics = evaluate(model, data, test_mask, weights, threshold=threshold_info["threshold"])
    return stats, test_metrics, best_iteration, threshold_info, history


def main() -> None:
    stats, test_metrics, best_iteration, threshold_info, history = train_model()

    if DOWNLOAD_ONLY:
        print("Dataset is ready.")
        return

    print("")
    print(format_summary(stats, test_metrics, best_iteration, threshold_info))

    if SHOW_PLOT:
        plot_training(history, test_metrics)


if __name__ == "__main__":
    main()
