import argparse
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data import prepare_data_split, prepare_dataloaders, train_sentencepiece_model, load_sentencepiece_model
from model import TransformerLM
from train import train
from generate import generate_submission

def main(config, output_file="submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load and preprocess dataset
    from datasets import load_dataset
    print("Loading dataset...")
    ptb = load_dataset("ptb-text-only/ptb_text_only", trust_remote_code=True)

    print("Training SentencePiece tokenizer...")
    train_sentencepiece_model(ptb['train'], vocab_size=config['vocab_size'])
    sp = load_sentencepiece_model("spm.model")

    # Update padding index and vocab size from SentencePiece model
    pad_idx = sp.pad_id()
    vocab_size = sp.vocab_size()
    print(f"BPE vocab size: {vocab_size}, PAD index: {pad_idx}")

    print("Preparing dataset splits...")
    train_data = prepare_data_split(ptb['train'], sp, config['max_len'], pad_idx)
    val_data = prepare_data_split(ptb['validation'], sp, config['max_len'], pad_idx)
    test_data = prepare_data_split(ptb['test'], sp, config['max_len'], pad_idx)

    print("Setting up DataLoaders...")
    train_loader, val_loader = prepare_dataloaders(train_data, val_data, config['batch_size'])
    test_loader = prepare_dataloaders(test_data, None, config['batch_size'])[0]

    # Step 2: Initialize model, optimizer, and loss function
    print("Initializing model...")
    model = TransformerLM(vocab_size, config['d_model'], config['n_head'], config['n_layer'], config['dropout']).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = CrossEntropyLoss(ignore_index=pad_idx)

    # Step 3: Train the model
    print("Starting training...")
    train(train_loader, val_loader, model, optimizer, loss_fn, config['num_epochs'], device, config['patience'], pad_idx)

    # Step 4: Test and generate submission
    print("Generating submission...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_ppl = generate_submission(test_loader, model, device, pad_idx, output_file)

    print(f"Test completed. Overall Test Perplexity: {test_ppl:.4f}. Submission saved to {output_file}.")


if __name__ == "__main__":
    # Example configuration
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

    parser = argparse.ArgumentParser(description="Run the Penn Treebank Language Modeling Task.")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="Output CSV file for submission.")
    args = parser.parse_args()

    main(config, output_file=args.output_file)
