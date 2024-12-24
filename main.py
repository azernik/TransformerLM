import argparse
import yaml
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pathlib import Path

from data import prepare_data_split, prepare_dataloaders, train_sentencepiece_model, load_sentencepiece_model
from model import TransformerLM
from train import train
from generate import generate_submission


def load_config(config_path="config.yaml", config_type="prod"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_type not in config:
        raise ValueError(f"Configuration type '{config_type}' not found in {config_path}")
    return config[config_type]


def main(config_path="config.yaml", config_type="run"):
    # Load the selected configuration
    config = load_config(config_path, config_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    paths = config['paths']
    output_dir = Path(paths['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and preprocess dataset
    from datasets import load_dataset
    print("Loading dataset...")
    ptb = load_dataset("ptb-text-only/ptb_text_only", trust_remote_code=True)

    print("Training SentencePiece tokenizer...")
    train_sentencepiece_model(
        ptb['train'],
        vocab_size=config['training']['vocab_size'],
        model_prefix=paths['spm_model'].split("/")[-1].split(".")[0],
        output_dir=output_dir,
    )

    print("Loading SentencePiece tokenizer...")
    sp = load_sentencepiece_model(paths['spm_model'])

    # Update padding index and vocab size
    pad_idx = sp.pad_id()
    vocab_size = sp.vocab_size()
    print(f"BPE vocab size: {vocab_size}, PAD index: {pad_idx}")

    print("Preparing dataset splits...")
    train_data = prepare_data_split(ptb['train'], sp, config['training']['max_len'], pad_idx)
    val_data = prepare_data_split(ptb['validation'], sp, config['training']['max_len'], pad_idx)
    test_data = prepare_data_split(ptb['test'], sp, config['training']['max_len'], pad_idx)

    print("Setting up DataLoaders...")
    train_loader, val_loader = prepare_dataloaders(train_data, val_data, config['training']['batch_size'])
    test_loader = prepare_dataloaders(test_data, None, config['training']['batch_size'])[0]

    # Step 2: Initialize model, optimizer, and loss function
    print("Initializing model...")
    model_config = config['model']
    model = TransformerLM(
        vocab_size, model_config['d_model'], model_config['n_head'],
        model_config['n_layer'], model_config['dropout']
    ).to(device)
    learning_rate = float(config['training']['learning_rate'])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=pad_idx)

    # Step 3: Train the model
    print("Starting training...")
    train(
        train_loader, val_loader, model, optimizer, loss_fn,
        config['training']['num_epochs'], device, config['training']['patience'],
        pad_idx, paths['model_checkpoint']
    )

    # Step 4: Test and generate submission
    print("Generating submission...")
    model.load_state_dict(torch.load(paths['model_checkpoint']))
    test_ppl = generate_submission(
        test_loader, model, device, pad_idx, output_file=paths['submission_file']
    )

    print(f"Test completed. Overall Test Perplexity: {test_ppl:.4f}. Submission saved to {paths['submission_file']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Penn Treebank Language Modeling Task.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--type", type=str, default="prod", help="Configuration type to use (e.g., 'dev' or 'prod').")
    args = parser.parse_args()

    main(config_path=args.config, config_type=args.type)