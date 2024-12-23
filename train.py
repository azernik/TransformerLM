import time
import torch
from torch.nn.utils import clip_grad_norm_

def train_one_epoch(loader, model, optimizer, loss_fn, device, pad_idx):
    model.train()
    total_loss = 0

    for batch in loader:
        inputs = batch['indices'].to(device)  # (batch_size, seq_len)
        masks = batch['masks'].to(device)    # (batch_size, seq_len)

        # Shift inputs and targets
        targets = inputs[:, 1:]
        inputs = inputs[:, :-1]

        # Forward pass
        logits = model(inputs, padding_mask=masks[:, :-1])
        logits = logits.view(-1, logits.size(-1))  # Flatten logits for loss computation
        targets = targets.reshape(-1)
        loss = loss_fn(logits, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def validate(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['indices'].to(device)
            masks = batch['masks'].to(device)

            targets = inputs[:, 1:]
            inputs = inputs[:, :-1]

            logits = model(inputs, padding_mask=masks[:, :-1])
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def train(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, device, patience, pad_idx):
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, pad_idx)
        val_loss = validate(val_loader, model, loss_fn, device)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Validation loss improved. Model saved.")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"Training completed. Best epoch: {best_epoch + 1} with Val Loss: {best_val_loss:.4f}.")
