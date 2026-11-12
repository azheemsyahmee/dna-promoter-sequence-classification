# DNA Promoter Sequence Classification with Bidirectional LSTM

[![CI Tests](https://github.com/azheemsyahmee/dna-promoter-sequence-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/azheemsyahmee/dna-promoter-sequence-classification/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework: TensorFlow](https://img.shields.io/badge/framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)

A deep learning framework for identifying prokaryotic promoter recognition motifs from raw DNA sequences, implementing Bidirectional Long Short-Term Memory (Bi-LSTM) networks and comparative recurrent architectures.

---

## 1. Problem Overview & Biological Context

In molecular biology, **promoters** are genomic regulatory elements located upstream of genes that dictate where transcription initiates. In *Escherichia coli*, transcription relies on RNA polymerase binding to conserved sequence signals (primarily the -35 hexamer and the -10 Pribnow box).

Because biological sequence regulation involves both upstream and downstream nucleotide context, unidirectional models often fail to capture bidirectional dependencies. This repository evaluates **Bidirectional LSTMs** to process genomic sequence data in forward (5' $\to$ 3') and reverse (3' $\to$ 5') orientations simultaneously.

---

## 2. Dataset Provenance & Preprocessing

* **Dataset:** Molecular Biology (Promoter Gene Sequences), UCI Machine Learning Repository
* **Sample Count:** 106 instances (53 promoters `+`, 53 non-promoters `-`)
* **Sequence Length:** 57 nucleotides per sequence (-50 to +7 relative to TSS)
* **One-Hot Representation:** Orthogonal 4D vector encoding per base:
  - `A`: `[1, 0, 0, 0]`
  - `C`: `[0, 1, 0, 0]`
  - `G`: `[0, 0, 1, 0]`
  - `T`: `[0, 0, 0, 1]`
* **Tensor Dimension:** Feature matrix $X \in \mathbb{R}^{106 \times 57 \times 4}$, label vector $y \in \{0, 1\}^{106}$.

---

## 3. Evaluated Model Architectures

The project investigates five neural sequence configurations:

1. **Experiment 1 (Bio-Aware Bi-LSTM - Primary):** `Bidirectional(LSTM(31))` $\to$ `Dropout(0.3)` $\to$ `Dense(32, tanh)` $\to$ `Dense(1, sigmoid)` with `RMSprop(lr=0.0031)`.
2. **Experiment 2 (Deep Memory Model):** Stacked `LSTM(31, return_sequences=True)` $\to$ `LSTM(31)` $\to$ `Dense(32, LeakyReLU)` with `Adamax(lr=0.0031)`.
3. **Experiment 3 (Fast Model):** Single `LSTM(31)` $\to$ `Dense(32, relu)` with `SGD(lr=0.01)`.
4. **Experiment 4 (Softmax Multi-Class):** Single `LSTM(31)` $\to$ `Dense(2, softmax)` with `Adam(lr=0.0031)` and sparse categorical cross-entropy.
5. **Experiment 5 (Regularized Model):** Single `LSTM(31)` $\to$ `Dropout(0.5)` $\to$ `Dense(32, tanh)` with `Adagrad(lr=0.001)`.

*Note: Architecture dimensions (`LSTM(31)`, `lr=0.0031`) reflect academic coursework parameterization linked to student identifier B23CS0031.*

---

## 4. Repository Structure

```
dna-promoter-sequence-classification/
├── .github/workflows/
│   └── tests.yml                     # Automated GitHub Actions CI workflow (Python 3.11)
├── data/
│   ├── README.md                     # Dataset details, citations, and biological schema
│   └── promoters.data                # Bundled UCI promoter dataset (106 sequences)
├── notebooks/
│   └── 01_promoter_bilstm_benchmark.ipynb # Consolidated reproducible research notebook
├── src/
│   ├── __init__.py
│   ├── data.py                       # Sequence parsing, cleaning, one-hot encoding, FASTA parser
│   ├── models.py                     # TensorFlow/Keras model definitions
│   ├── train.py                      # Training routines with early stopping
│   └── evaluate.py                   # Performance metrics and inference helpers
├── tests/
│   ├── __init__.py
│   ├── test_data.py                  # Data loading, shape, and FASTA parser tests
│   ├── test_evaluate.py              # Metric calculation tests
│   ├── test_models.py                # Model registry and compilation tests
│   └── run_tests.py                  # Standalone test runner
├── .gitignore
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 5. Quick Start & Reproduction

### Installation
```bash
git clone https://github.com/azheemsyahmee/dna-promoter-sequence-classification.git
cd dna-promoter-sequence-classification
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Automated Tests
```bash
python tests/run_tests.py
# or via unittest discovery:
python -m unittest discover -s tests -v
```

### Sequence Inference Example
```python
from src.data import one_hot_encode_sequence
from src.models import build_bilstm_model

# 1. Build model architecture
model = build_bilstm_model(input_shape=(57, 4))

# 2. Encode arbitrary 57-bp sequence
sequence = "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt"
x_input = one_hot_encode_sequence(sequence).reshape(1, 57, 4)

# 3. Predict promoter probability
prob = model.predict(x_input)[0][0]
print(f"Promoter Probability: {prob:.4f}")
```

---

## 6. Academic Attribution & License

* **Institution:** Universiti Teknologi Malaysia (UTM)
* **Course:** SECB 3223 — Computational Biology 2
* **Author:** Azheem Syahmee ([@azheemsyahmee](https://github.com/azheemsyahmee))
* **License:** [MIT License](LICENSE)
