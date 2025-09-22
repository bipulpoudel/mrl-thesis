# =============================================================================
# 1. IMPORTS
# =============================================================================
# --- Standard Python Libraries ---
import os
import copy  # Used to deepcopy the model's state for saving the best version

# --- Core Machine Learning Libraries ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import pandas as pd # For creating clean summary tables of results
from tqdm import tqdm # For creating elegant progress bars

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Custom Project Files (assumed to be in the same directory or accessible) ---
from resnet_18_model import ResNet18_CIFAR
from load_data_tasks import load_data_cifar10_by_tasks, load_data_cifar100_by_tasks, load_data_split_mnist_by_tasks
from logger import logger
from utils import optimize_device_for_pytorch, set_reproducibility_seeds


# =============================================================================
# 2. HYPERPARAMETERS CONFIGURATION
# =============================================================================
class Hyperparameters:
    """
    A dedicated class to centralize all hyperparameters for the experiment.
    This makes it easy to see and modify the configuration in one place.
    """
    # --- Training Duration ---
    # Number of epochs to train per "task" in the dataset.
    # For joint training, all tasks are combined, so total epochs = EPOCHS_PER_TASK * num_tasks.
    EPOCHS_PER_TASK = 20
    
    # --- Reproducibility ---
    # A fixed seed ensures that random operations (like weight initialization, data shuffling,
    # and validation splits) are the same every time, making results reproducible.
    SEED = 42
    
    # --- Data Loading ---
    BATCH_SIZE = 32     # Number of samples processed in one forward/backward pass.
    NUM_WORKERS = 0     # Number of subprocesses for data loading. 0 means data is loaded in the main process.
    
    # --- Optimizer Settings (for Stochastic Gradient Descent - SGD) ---
    LEARNING_RATE = 0.001 # Initial learning rate. Controls how much we adjust model weights.
    MOMENTUM = 0.9       # Helps the optimizer to continue in the correct direction and dampens oscillations.
    WEIGHT_DECAY = 5e-4  # L2 regularization term. Helps prevent overfitting by penalizing large weights.

    # --- Regularization ---
    # If validation accuracy doesn't improve for this many consecutive epochs, stop training.
    EARLY_STOPPING_PATIENCE = 10


# =============================================================================
# 3. UTILITY FUNCTIONS (Plotting & Evaluation)
# =============================================================================

def plot_and_save_metrics(history, dataset_name, save_dir='plots'):
    """
    Plots training & validation loss and accuracy from the training history
    and saves the generated plot to a file.

    Args:
        history (dict): A dictionary containing lists of metrics for each epoch.
                        Expected keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        dataset_name (str): The name of the dataset (e.g., 'split_cifar10') for titles.
        save_dir (str): The directory where the plot image will be saved.
    """
    # Ensure the output directory exists. If not, create it.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created directory: {save_dir}")

    # Create a list of epoch numbers for the x-axis of our plots.
    epochs = range(1, len(history['train_loss']) + 1)

    # Create a figure with two subplots, one for loss and one for accuracy.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f'Joint Training Metrics for {dataset_name.upper()}', fontsize=16)

    # --- Subplot 1: Loss ---
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Subplot 2: Accuracy ---
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout to prevent the main title from overlapping with subplots.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure to a PNG file.
    save_path = os.path.join(save_dir, f'joint_training_{dataset_name}.png')
    plt.savefig(save_path)
    plt.close() # Close the figure to free up memory, important when running many experiments.
    logger.info(f"Metrics plot saved to {save_path}")


def evaluate_model(model, device, data_loader, criterion):
    """
    Evaluates the model's performance on a given dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        device (torch.device): The device (CPU or CUDA) to run evaluation on.
        data_loader (DataLoader): DataLoader for the dataset to evaluate (e.g., validation or test).
        criterion: The loss function (e.g., CrossEntropyLoss).

    Returns:
        tuple: A tuple containing the average loss and accuracy.
    """
    # Set the model to evaluation mode. This is crucial as it disables layers like
    # Dropout and uses the running statistics for Batch Normalization.
    model.eval()
    
    # Initialize counters for loss and correct predictions.
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Use torch.no_grad() to disable gradient calculations. This speeds up inference
    # and reduces memory usage since we are not training.
    with torch.no_grad():
        # Loop through all batches in the data loader.
        for images, labels, _ in tqdm(data_loader, desc="Evaluating", leave=False):
            # Move data to the selected device.
            images, labels = images.to(device), labels.to(device)
            
            # Perform a forward pass.
            outputs = model(images)
            
            # Calculate the loss for the current batch.
            loss = criterion(outputs, labels)
            
            # Accumulate the total loss.
            total_loss += loss.item()
            
            # Get the predicted class by finding the index of the max log-probability.
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total sample count and correct prediction count.
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    # Calculate the average loss and accuracy for the entire dataset.
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples
    
    return avg_loss, accuracy


# =============================================================================
# 4. MAIN EXPERIMENT LOGIC
# =============================================================================

def run_experiment(dataset_name, device):
    """
    Runs the full joint training experiment for a single specified dataset.
    This includes data loading, creating a validation split, training with
    early stopping, final evaluation on the test set, and plotting results.
    """
    # --- 4.1: Load and Prepare Data ---
    logger.info(f"Loading and combining datasets for {dataset_name}...")
    if dataset_name == "split_cifar10":
        train_datasets, test_datasets, num_classes, _ = load_data_cifar10_by_tasks(return_datasets=True)
    elif dataset_name == "split_cifar100":
        train_datasets, test_datasets, num_classes, _ = load_data_cifar100_by_tasks(return_datasets=True)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
        
    total_epochs = Hyperparameters.EPOCHS_PER_TASK * len(train_datasets)

    # --- 4.2: Create a Validation Split ---
    # To get an unbiased measure of performance, we must not use the test set for
    # any training decisions (like early stopping). We split the original training
    # data into a new, smaller training set and a validation set.
    joint_train_dataset = ConcatDataset(train_datasets)
    
    val_size = int(0.1 * len(joint_train_dataset)) # Use 10% of training data for validation
    train_size = len(joint_train_dataset) - val_size
    
    # Use a fixed generator for the random split to ensure the split is the same every time.
    generator = torch.Generator().manual_seed(Hyperparameters.SEED)
    train_subset, val_subset = random_split(joint_train_dataset, [train_size, val_size], generator=generator)
    
    logger.info("=" * 60)
    logger.info(f"STARTING: JOINT TRAINING (UPPER BOUND) - {dataset_name.upper()}")
    logger.info(f"Total Epochs: {total_epochs}, Patience: {Hyperparameters.EARLY_STOPPING_PATIENCE}")
    logger.info(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    logger.info("=" * 60)

    # --- 4.3: Create DataLoaders ---
    # DataLoaders handle batching, shuffling, and multi-threaded data loading.
    joint_test_dataset = ConcatDataset(test_datasets)
    use_pin_memory = device.type == 'cuda' # Speeds up CPU-to-GPU data transfer.
    
    train_loader = DataLoader(train_subset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=True, num_workers=Hyperparameters.NUM_WORKERS, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_subset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=False, num_workers=Hyperparameters.NUM_WORKERS, pin_memory=use_pin_memory)
    test_loader = DataLoader(joint_test_dataset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=False, num_workers=Hyperparameters.NUM_WORKERS, pin_memory=use_pin_memory)

    # --- 4.4: Initialize Model, Loss, Optimizer, and Scheduler ---
    model = ResNet18_CIFAR(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss() # Standard loss function for multi-class classification.
    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY)
    
    # The scheduler adjusts the learning rate at specific points during training, which often
    # leads to better convergence and final performance.
    milestones = [int(0.6 * total_epochs), int(0.8 * total_epochs)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # --- 4.5: Training Loop ---
    # Initialize variables to track the best model and early stopping criteria.
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(total_epochs):
        # Set model to training mode
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        
        # --- Batch Loop ---
        for i, (images, labels, _) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # The core training steps:
            optimizer.zero_grad()      # 1. Clear previous gradients.
            outputs = model(images)    # 2. Forward pass.
            loss = criterion(outputs, labels) # 3. Calculate loss.
            loss.backward()            # 4. Backward pass (compute gradients).
            optimizer.step()           # 5. Update model weights.
            
            # Update running metrics for logging.
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=f'{running_loss / (i + 1):.4f}')

        # --- Post-Epoch Evaluation & Logic ---
        # Calculate epoch-level training metrics.
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        
        # Evaluate on the VALIDATION set.
        val_loss, val_acc = evaluate_model(model, device, val_loader, criterion)
        
        logger.info(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Store metrics for plotting later.
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # --- Early Stopping and Best Model Saving ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Use deepcopy to save a snapshot of the model's weights at its best performance.
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            logger.info(f"  -> New best validation accuracy: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= Hyperparameters.EARLY_STOPPING_PATIENCE:
            logger.warning(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break # Exit the training loop.
        
        # Step the learning rate scheduler.
        scheduler.step()

    # --- 4.6: Final Evaluation on the Test Set ---
    logger.info("-" * 60)
    logger.info("Training complete. Loading best model state for final evaluation.")
    
    # Load the best model weights found during training (based on validation performance).
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Evaluating final model on the UNSEEN test set...")
    final_test_loss, final_test_acc = evaluate_model(model, device, test_loader, criterion)
    
    logger.info(f"FINAL UNBIASED TEST ACCURACY: {final_test_acc:.2f}%")
    logger.info("-" * 60)
    
    # Generate and save the training/validation plot.
    plot_and_save_metrics(history, dataset_name)
    
    # Return the final, unbiased test accuracy.
    return final_test_acc


# =============================================================================
# 5. SCRIPT ENTRY POINT
# =============================================================================

def main():
    """
    Main function to orchestrate the entire experiment across multiple datasets.
    """
    # --- Global Setup ---
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    # --- Run Experiments ---
    datasets_to_run = ["split_cifar10","split_cifar100"]
    results = []

    for dataset_name in datasets_to_run:
        best_acc = run_experiment(dataset_name, device)
        results.append({
            "Dataset": dataset_name,
            "Best Test Accuracy (%)": f"{best_acc:.2f}"
        })

    # --- Final Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("           JOINT TRAINING OVERALL RESULTS (UPPER BOUND)")
    logger.info("=" * 60)
    results_df = pd.DataFrame(results)
    
    # Log the results table to the log file.
    logger.info("\n" + results_df.to_string(index=False))

    # Also print the results table to the console for immediate visibility.
    print("\n" + "=" * 60)
    print("           JOINT TRAINING OVERALL RESULTS (UPPER BOUND)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

# This standard Python construct ensures that the main() function is called
# only when this script is executed directly (not when imported as a module).
if __name__ == '__main__':
    main()