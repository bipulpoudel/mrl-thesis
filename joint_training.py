import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import pandas as pd

# Import from existing project files
from resnet_18_model import ResNet18_CIFAR
from load_data_tasks import load_data_cifar10_by_tasks, load_data_cifar100_by_tasks, load_data_split_mnist_by_tasks
from logger import logger
from utils import optimize_device_for_pytorch, set_reproducibility_seeds

class Hyperparameters:
    """Hyperparameters for Joint Training."""
    # Joint training is performed on the entire dataset, so more epochs are needed for convergence.
    EPOCHS_PER_TASK = 5
    
    # Seed for reproducibility
    SEED = 42
    
    # A larger batch size is often feasible and more efficient for standard training.
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    
    # Optimizer settings - a higher initial learning rate with a scheduler is common for CIFAR.
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

def evaluate_model(model, device, test_loader, criterion):
    """Evaluates the model on the provided test data loader."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def run_experiment(dataset_name, device):
    """
    Runs the full joint training experiment for a single specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to run ("split_cifar10", "split_cifar100", "split_mnist").
        device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
        
    Returns:
        float: The best test accuracy achieved during training.
    """
    # --- Load and Combine Data ---
    logger.info(f"Loading and combining datasets for {dataset_name}...")
    if dataset_name == "split_cifar10":
        train_datasets, test_datasets, num_classes, _ = load_data_cifar10_by_tasks(
            return_datasets=True
        )
    elif dataset_name == "split_cifar100":
        train_datasets, test_datasets, num_classes, _ = load_data_cifar100_by_tasks(
            return_datasets=True
        )
    elif dataset_name == "split_mnist":
        train_datasets, test_datasets, num_classes, _ = load_data_split_mnist_by_tasks(
            return_datasets=True
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
        
    total_epochs = Hyperparameters.EPOCHS_PER_TASK * len(train_datasets)

    logger.info("=" * 60)
    logger.info(f"STARTING: JOINT TRAINING (UPPER BOUND) - {dataset_name.upper()}")
    logger.info(f"Total Epochs: {total_epochs}")
    logger.info(f"Learning Rate: {Hyperparameters.LEARNING_RATE}")
    logger.info(f"Batch Size: {Hyperparameters.BATCH_SIZE}")
    logger.info(f"Number of Classes: {num_classes}")
    logger.info("=" * 60)

    # Use ConcatDataset to merge all task-specific datasets into one
    joint_train_dataset = ConcatDataset(train_datasets)
    joint_test_dataset = ConcatDataset(test_datasets)

    logger.info(f"Total training samples: {len(joint_train_dataset)}")
    logger.info(f"Total test samples: {len(joint_test_dataset)}")

    use_pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        joint_train_dataset,
        batch_size=Hyperparameters.BATCH_SIZE,
        shuffle=True,
        num_workers=Hyperparameters.NUM_WORKERS,
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        joint_test_dataset,
        batch_size=Hyperparameters.BATCH_SIZE,
        shuffle=False,
        num_workers=Hyperparameters.NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    # --- Model, Loss, Optimizer ---
    # NOTE: Using the same ResNet18 for all datasets for consistency.
    # Data loaders are expected to handle transformations (e.g., MNIST to 3-channel).
    model = ResNet18_CIFAR(num_classes=num_classes, pretrained=False) # Pre-trained weights might not be ideal for MNIST
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Hyperparameters.LEARNING_RATE,
        momentum=Hyperparameters.MOMENTUM,
        weight_decay=Hyperparameters.WEIGHT_DECAY
    )
    
    # A learning rate scheduler is crucial for good performance.
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*total_epochs), int(0.8*total_epochs)], gamma=0.1)

    # --- Training Loop ---
    best_test_acc = 0.0
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [{dataset_name}]", leave=False)
        
        for i, (images, labels, _) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_acc = 100 * correct_train / total_train
            current_loss = running_loss / (i + 1)
            progress_bar.set_postfix(loss=f'{current_loss:.4f}', acc=f'{train_acc:.2f}%')

        # Evaluate on the test set after each epoch
        test_loss, test_acc = evaluate_model(model, device, test_loader, criterion)
        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Train Loss: {current_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )   
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            logger.info(f"New best test accuracy: {best_test_acc:.2f}%")
        
        scheduler.step()

    logger.info("-" * 60)
    logger.info(f"JOINT TRAINING FOR {dataset_name.upper()} COMPLETED")
    logger.info(f"FINAL (BEST) TEST ACCURACY: {best_test_acc:.2f}%")
    logger.info("-" * 60)
    
    return best_test_acc

def main():
    """Main function to run the joint training experiment across all datasets."""
    
    # --- Global Setup ---
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    datasets_to_run = ["split_mnist", "split_cifar10", "split_cifar100"]
    results = []

    for dataset_name in datasets_to_run:
        best_acc = run_experiment(dataset_name, device)
        results.append({
            "Dataset": dataset_name,
            "Best Test Accuracy (%)": f"{best_acc:.2f}"
        })

    # --- Final Summary Table ---
    logger.info("\n" + "=" * 60)
    logger.info("           JOINT TRAINING OVERALL RESULTS (UPPER BOUND)")
    logger.info("=" * 60)
    
    # Using pandas to create a clean table.
    results_df = pd.DataFrame(results)
    
    # For logger output
    logger.info("\n" + results_df.to_string(index=False))

    # For console-friendly printing
    print("\n" + "=" * 60)
    print("           JOINT TRAINING OVERALL RESULTS (UPPER BOUND)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    main()