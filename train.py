import time
import torch
from torch.nn.utils import clip_grad_norm_

def train_one_epoch(loader, model, optimizer, loss_fn, device, pad_idx, epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        inputs = batch['indices'].to(device)
        masks = batch['masks'].to(device)

        # Shift inputs and targets
        targets = inputs[:, 1:]
        inputs = inputs[:, :-1]

        # Forward pass
        logits = model(inputs, padding_mask=masks[:, :-1])
        logits = logits.view(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = loss_fn(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 500 == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate(loader, model, loss_fn, device, epoch=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = batch['indices'].to(device)
            masks = batch['masks'].to(device)

            targets = inputs[:, 1:]
            inputs = inputs[:, :-1]

            logits = model(inputs, padding_mask=masks[:, :-1])
            logits = logits.view(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    if epoch is not None:
        print(f"Validation Loss after Epoch [{epoch}]: {avg_loss:.4f}")
    return avg_loss


def train(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, device, patience, pad_idx, checkpoint_path):
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, pad_idx, epoch + 1)
        val_loss = validate(val_loader, model, loss_fn, device, epoch + 1)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Model saved at {checkpoint_path}.")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"\nTraining completed. Best Epoch: {best_epoch + 1} with Validation Loss: {best_val_loss:.4f}")
