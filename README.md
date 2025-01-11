# Transformer Language Model on Penn Treebank

## Overview
This project implements a Transformer-based Language Model trained on the Penn Treebank dataset. It focuses on predicting the next token in a sequence using subword tokenization (SentencePiece BPE) and evaluates performance using token-level and sentence-level perplexity.

## Features
- **Transformer Architecture**: Decoder-only Transformer with learned positional encodings.
- **Tokenization**: Byte Pair Encoding (BPE) using SentencePiece to reduce vocabulary sparsity.
- **Evaluation Metrics**: Token-level perplexity and sentence-level perplexity.
- **Training Features**: Early stopping, gradient clipping, mixed-precision training (AMP), and configurable hyperparameters.
- **Parallelized Training**: Multi-GPU support for faster convergence.

## Results
| Model Version | Token PPL | Sentence PPL | Validation Loss |
|---------------|-----------|--------------|-----------------|
| Baseline      | 49.34     | 278.03       | 3.89            |
| BPE           | 44.26     | 22.66        | 2.97            |

Plots and detailed comparisons can be found in the linked [report](#report).

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.5+
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
   python main.py --config config.yaml --type prod
   ```

2. **Generate Submission File**:
   The script will output a CSV file containing sentence-level perplexity scores for submission.

### Configurations
Model and training hyperparameters can be adjusted in `config.yaml`. Example:
```yaml
prod:
  model:
    d_model: 512
    n_head: 8
    n_layer: 6
    dropout: 0.3
  training:
    learning_rate: 5e-5
    batch_size: 64
    num_epochs: 50
    patience: 10
    max_len: 50
    vocab_size: 2000
  paths:
    spm_model: prod_models/spm.model
    train_text: prod_models/train.txt
    output_dir: prod_models
    model_checkpoint: prod_models/best_model.pth
    submission_file: prod_submission.csv
```

### Updates to Training
This project now uses mixed-precision training (AMP) and multi-GPU support. Ensure you have CUDA-enabled GPUs. The key updates include:
- **Mixed Precision Training**: Automatically handled by `torch.amp.autocast("cuda")` for improved speed and memory efficiency.
- **Multi-GPU Support**: Automatically wraps the model using `torch.nn.DataParallel` when multiple GPUs are available.
