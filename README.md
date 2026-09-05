# Elliptic Bitcoin GCN

A graph neural network project for classifying Bitcoin transactions as licit or illicit using a Graph Convolutional Network (GCN) implemented with PyTorch Geometric.

The project treats Bitcoin transactions as graph nodes and payment flows as directed edges. It trains a two-layer GCN to use both transaction features and graph connectivity.

## Overview

The Elliptic Bitcoin dataset contains anonymized Bitcoin transactions collected across 49 time steps.

Each graph node represents a transaction, while each edge represents a Bitcoin payment flow between transactions. The dataset includes known licit transactions, known illicit transactions, and transactions with unknown labels.

This project:

- loads and preprocesses the Elliptic transaction graph;
- trains a two-layer GCN classifier;
- creates train, validation, and test masks;
- normalizes node features using training data only;
- handles class imbalance with weighted cross-entropy loss;
- applies early stopping based on validation F1-score for illicit transactions;
- evaluates the final model with classification and ranking metrics;
- visualizes training loss, validation F1-score, and the confusion matrix.

## Dataset

The project uses the **Elliptic Data Set**, published by Elliptic.

- Official dataset page: [Elliptic Data Set on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Dataset description: [Elliptic Data Set announcement](https://www.elliptic.co/insights/elliptic-dataset-cryptocurrency-financial-crime/)
- Research paper: [Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics](https://arxiv.org/abs/1908.02591)

The raw dataset contains:

- 203,769 transaction nodes;
- 234,355 directed payment-flow edges;
- 166 anonymized transaction features;
- 4,545 illicit transactions;
- 42,019 licit transactions;
- transactions with unknown labels.

### Labels

The original dataset uses the following labels:

| Original label | Meaning |
|---|---|
| `1` | Illicit transaction |
| `2` | Licit transaction |
| `unknown` | Transaction without a known class |

This project remaps labeled transactions to:

| Model label | Meaning |
|---|---|
| `0` | Licit transaction |
| `1` | Illicit transaction |

Unknown transactions remain in the graph but are excluded from supervised training and evaluation.

## Download the Dataset

### Recommended: automatic download

The project uses PyTorch Geometric's `EllipticBitcoinDataset`. During the first run, PyTorch Geometric downloads and extracts the required files automatically.

Run:

```bash
python train_gcn.py
```

The files are stored in:

```text
data/elliptic/
```

PyTorch Geometric downloads these files into its dataset cache:

```text
elliptic_txs_features.csv
elliptic_txs_edgelist.csv
elliptic_txs_classes.csv
```

The PyTorch Geometric dataset implementation downloads these files from its maintained dataset mirror and processes them into a graph object. See the [EllipticBitcoinDataset source code](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/elliptic.py).

### Manual download

Use this option if the automatic download fails or if you want to inspect the original CSV files.

1. Open the [Elliptic Data Set on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
2. Sign in to Kaggle and accept the dataset terms if requested.
3. Click **Download**.
4. Extract the downloaded archive.
5. Create this folder structure in the project:

```text
data/
└── elliptic/
    └── raw/
        ├── elliptic_txs_features.csv
        ├── elliptic_txs_edgelist.csv
        └── elliptic_txs_classes.csv
```

Do not add an additional nested folder such as:

```text
data/elliptic/elliptic/raw/
```

The dataset loader expects the three CSV files directly inside `data/elliptic/raw/`.

## Dataset License

The Elliptic dataset is distributed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International** license.

The dataset is not included in this repository. Download it from the official Kaggle page and use it according to its license terms.

## Requirements

- Python 3.10 or newer
- PyTorch
- PyTorch Geometric
- NumPy
- scikit-learn
- Matplotlib

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Environment Setup

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Training

Run the command from the project root:

```bash
python train_gcn.py
```

The script prints:

- dataset statistics;
- training loss;
- validation F1-score;
- early-stopping information;
- best validation iteration;
- final test metrics.

It also displays plots for training loss, validation F1-score, and the final confusion matrix.

## Model

The classifier uses two graph convolution layers:

```text
Transaction features
        ↓
GCNConv → ReLU → Dropout
        ↓
GCNConv
        ↓
Licit or illicit prediction
```

The model aggregates information from neighboring transactions, allowing it to use transaction relationships in addition to individual node features.

## Configuration

Edit the constants at the top of `train_gcn.py` to change the experiment settings.

| Setting | Default value | Purpose |
|---|---:|---|
| `ITERATIONS` | `200` | Maximum number of training iterations |
| `HIDDEN_CHANNELS` | `128` | Width of the hidden GCN layer |
| `DROPOUT` | `0.5` | Dropout probability |
| `LEARNING_RATE` | `0.01` | Adam optimizer learning rate |
| `VALIDATION_SIZE` | `0.15` | Validation share of labeled training nodes |
| `DECISION_THRESHOLD` | `0.65` | Probability threshold for the illicit class |
| `PATIENCE` | `8` | Early-stopping patience |
| `MAKE_UNDIRECTED` | `True` | Converts directed edges to undirected edges |
| `DEVICE` | `"cpu"` | Training device |

Use `"cuda"` for an NVIDIA GPU or `"mps"` for Apple Silicon when supported:

```python
DEVICE = "mps"
```

## Evaluation

The project reports:

- best validation iteration;
- decision threshold;
- accuracy;
- balanced accuracy;
- precision, recall, and F1-score;
- illicit-class F1-score;
- ROC-AUC;
- average precision;
- confusion matrix in the format `[[TN, FP], [FN, TP]]`.

The illicit class represents a minority of labeled transactions. Accuracy alone can therefore be misleading, so the project also reports balanced accuracy, recall, F1-score, ROC-AUC, and average precision.

## Project Structure

```text
.
├── train_gcn.py             # Dataset loading, model, training, and evaluation
├── requirements.txt         # Python dependencies
├── tests/
│   └── test_train_gcn.py    # Unit tests
├── data/
│   └── elliptic/            # Downloaded dataset cache, ignored by Git
├── .gitignore
└── README.md
```

## Notes

This repository contains an academic machine-learning experiment.

The model output does not prove that a real transaction or entity is lawful, unlawful, suspicious, or linked to criminal activity. Do not use it for financial, legal, compliance, or law-enforcement decisions.
