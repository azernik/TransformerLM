import torch
import pandas as pd
from torch.nn import CrossEntropyLoss


def generate_submission(test_loader, model, device, pad_idx, output_file="submission.csv"):
    """
    Generate a submission file and calculate overall test perplexity.
    """
    model.eval()
    results = []
    total_loss = 0
    total_tokens = 0

    loss_fn = CrossEntropyLoss(ignore_index=pad_idx, reduction='none')

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # Generate IDs for each sentence
            start_idx = idx * test_loader.batch_size
            ids = list(range(start_idx, start_idx + len(batch['indices'])))

            # Move inputs and masks to device
            inputs = batch['indices'].to(device)
            masks = batch['masks'].to(device)

            # Shift inputs and targets
            targets = inputs[:, 1:]
            inputs = inputs[:, :-1]

            # Forward pass
            logits = model(inputs, padding_mask=masks[:, :-1])
            logits = logits.reshape(-1, logits.size(-1))

            # Compute token-level loss
            token_losses = loss_fn(logits, targets.reshape(-1))
            total_loss += token_losses.sum().item()

            # Count valid tokens
            valid_tokens = (targets != pad_idx).sum().item()
            total_tokens += valid_tokens

            # Compute sentence-level perplexity
            token_losses = token_losses.view(inputs.size(0), -1)  # Reshape to (batch_size, seq_len - 1)
            sentence_losses = token_losses.sum(dim=1)
            sentence_lengths = (targets != pad_idx).sum(dim=1)
            sentence_perplexities = torch.exp(sentence_losses / sentence_lengths)

            # Collect results
            results.extend([{"ID": id_, "ppl": ppl.item()} for id_, ppl in zip(ids, sentence_perplexities)])

    # Calculate overall test perplexity
    overall_ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Test Perplexity (Overall): {overall_ppl.item():.4f}")

    # Save to CSV
    submission_df = pd.DataFrame(results)
    submission_df[['ID', 'ppl']].to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")
    return overall_ppl
