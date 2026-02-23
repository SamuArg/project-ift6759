
import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# seed experiment for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def plot_metrics(train_losses, valid_losses, train_accs, valid_accs, model_name, figdir):
    """Generates and saves training curves for loss and accuracy."""
    #TODO adjust figure parameters to make them pretty

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(valid_losses, label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Accuracy
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(valid_accs, label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    fig_path = os.path.join(figdir, f'learning_curves_{model_name}.png')
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Figures saved to {fig_path}")

def run_epoch(model, dataloader, loss_fn, accuracy_fn, device, optimizer=None, is_training=True):
    """
    Handles both training and evaluation iterations over a dataloader.
    If optimizer is provided and is_training is True, updates model weights.
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0

    # Use torch.set_grad_enabled to turn off gradients for validation/testing
    with torch.set_grad_enabled(is_training):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_training:
                optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Note: Depending on the SeisBench model, 'outputs' might be a tuple or dict. 
            # You may need to unpack it here depending on the specific model architecture.
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            if is_training:
                loss.backward()
                optimizer.step()

            # Calculate accuracy and accumulate
            acc = accuracy_fn(outputs, targets)
            
            # Multiply by batch size to account for variable-sized last batches
            total_loss += loss.item() * inputs.size(0)
            total_acc += acc * inputs.size(0)

    # Calculate epoch averages
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_acc / len(dataloader.dataset)
    
    return avg_loss, avg_acc
    
def train(
        model: nn.Module, 
        train_set: DataLoader, 
        validation_set: DataLoader, 
        test_set: DataLoader, 
        device: torch.device, 
        loss_fn, 
        accuracy_fn, 
        logdir: str = None, 
        figdir: str = None, 
        modeldir: str = None,
        optimizer: optim.Optimizer = None,
        epochs: int = 500, 
        print_every: int = 10
    ):
    """
    Main training loop.
    """
    model = model.to(device)
    model_name = model.__class__.__name__
    
    # Ensure an optimizer is provided, otherwise default to Adam with lr=1e-3
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    
    best_valid_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    print(f'Training {model_name} model on {device}...')
    start_time = time.time()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        
        # 1. Train Step
        train_loss, train_acc = run_epoch(
            model, train_set, loss_fn, accuracy_fn, device, optimizer=optimizer, is_training=True
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 2. Validation Step
        valid_loss, valid_acc = run_epoch(
            model, validation_set, loss_fn, accuracy_fn, device, is_training=False
        )
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # 3. Track best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_weights = copy.deepcopy(model.state_dict())

        # 4. Logging
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            tqdm.write(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc:.4f}"
            )

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"===== Best Validation Accuracy: {best_valid_acc:.4f} =====")

    # Load best model weights for testing
    model.load_state_dict(best_model_weights)

    # 5. Test Step
    print(f"\nEvaluating {model_name} on Test Set...")
    test_loss, test_acc = run_epoch(
        model, test_set, loss_fn, accuracy_fn, device, is_training=False
    )
    print(f"===== Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} =====")

    # 6. Save Logs
    if logdir is not None:
        print(f'\nWriting training logs to {logdir}...')
        os.makedirs(logdir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(logdir, f'results_{model_name}.json'), 'w') as f:
            json.dump({
                "model": model_name,
                "epochs": epochs,
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_accs": train_accs,
                "valid_accs": valid_accs,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, f, indent=4)
            
    # Save best model weights
    if modeldir is not None:
        torch.save(best_model_weights, os.path.join(modeldir, f'best_{model_name}.pth'))

    # 7. Save Figures
    if figdir is not None:
        plot_metrics(train_losses, valid_losses, train_accs, valid_accs, model_name, figdir)

    return model, {"test_loss": test_loss, "test_acc": test_acc}