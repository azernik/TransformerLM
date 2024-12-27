# Transformer Language Model on Penn Treebank

## Overview
This project implements a Transformer-based Language Model trained on the Penn Treebank dataset. It focuses on predicting the next token in a sequence using subword tokenization (SentencePiece BPE) and evaluates performance using token-level and sentence-level perplexity.

Note: For a detailed explanation of the experiments and results, see the accompanying [report](https://drive.google.com/file/d/1qExlqxQSto_HKaiWWVw0KoLp83qzYdEt/view?usp=drive_link).

## Features
- **Transformer Architecture**: Decoder-only Transformer with learned positional encodings.
- **Tokenization**: Byte Pair Encoding (BPE) using SentencePiece to reduce vocabulary sparsity.
- **Evaluation Metrics**: Token-level perplexity and sentence-level perplexity.
- **Training Features**: Early stopping, gradient clipping, and configurable hyperparameters.

## Results
| Model Version | Token PPL | Sentence PPL | Validation Loss |
|---------------|-----------|--------------|-----------------|
| Baseline      | 49.34     | 278.03       | 3.89            |
| BPE           | 44.26     | 22.66        | 2.97            |

Plots and detailed comparisons can be found in the linked [report](#report).

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset
The project uses the Penn Treebank dataset, which can be downloaded from Hugging Face:
```bash
pip install datasets
```

### Usage
1. **Train and Evaluate the Model**:
   ```bash
   python main.py --output_file submission.csv
   ```

2. **Generate Submission File**:
   The script will output a CSV file containing sentence-level perplexity scores for submission.

### Configurations
Model and training hyperparameters can be adjusted in `main.py` via the `config` dictionary:
```python
config = {
    "d_model": 512,
    "n_head": 8,
    "n_layer": 6,
    "dropout": 0.3,
    "learning_rate": 5e-5,
    "batch_size": 64,
    "num_epochs": 50,
    "patience": 10,
    "max_len": 50,
    "vocab_size": 2000,
}
```

### Results
Pre-trained model checkpoints and submission examples are included in the repository.

### References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Subword Regularization](https://arxiv.org/abs/1804.10959)
