import torch
import torch.nn as nn
from tqdm import tqdm
from resnet_18_model import ResNet18_CIFAR
from load_data_tasks import load_data_cifar10_by_tasks, load_data_cifar100_by_tasks, load_data_split_mnist_by_tasks
from logger import logger
from utils import optimize_device_for_pytorch, set_reproducibility_seeds
import torch.optim as optim
import random
import sys
import numpy as np

class Hyperparameters:
    # Increased epochs for better convergence
    EPOCHS_PER_TASK = 5 
    
    #Seed
    SEED = 42
    
    #Data directory
    DATA_DIR = "./data"
    
    #Batch sizes and number of workers
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    #Learning rate
    LEARNING_RATE = 0.01
    
    #Momentum
    MOMENTUM = 0.9
    
    #Weight decay
    WEIGHT_DECAY = 5e-4 

# (The ReservoirBuffer, evaluate_model, and train_model functions remain unchanged)

class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.n_seen_so_far = 0

    def add_to_buffer(self, items, labels):
        """
        Adds samples to the buffer using reservoir sampling.
        """
        for item, label in zip(items, labels):
            sample = (item.detach().cpu(), label.detach().cpu())
            
            if self.n_seen_so_far < self.buffer_size:
                self.buffer.append(sample)
            else:
                j = random.randint(0, self.n_seen_so_far)
                if j < self.buffer_size:
                    self.buffer[j] = sample
            
            self.n_seen_so_far += 1

    def get_buffer_size(self):
        return len(self.buffer)
    
    def get_samples_from_buffer(self, batch_size, device):
        if len(self.buffer) == 0:
            return None, None
        num_samples_to_get = min(len(self.buffer), batch_size)
        random_samples = random.sample(self.buffer, num_samples_to_get)
        items, labels = zip(*random_samples)
        items_tensor = torch.stack(items).to(device)
        labels_tensor = torch.stack(labels).to(device)
        return items_tensor, labels_tensor

def evaluate_model(model, device, validation_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for i, (images, labels, task_id) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)   
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss / len(validation_loader)
    val_acc = 100 * val_correct / val_total
    return val_loss, val_acc

def train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, 
                epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=0):
    tqdm_epochs = tqdm(range(epochs), desc=f"Training Task {task_id+1}")
    for epoch in tqdm_epochs:
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']
        tqdm_train_loader = tqdm(train_loader, desc="Progress:", leave=False)
        for i, (images, labels, task_id_batch) in enumerate(tqdm_train_loader):
            current_images, current_labels = images.to(device), labels.to(device)
            current_batch_size = current_images.size(0)
            
            total_images, total_labels = current_images, current_labels
            if reservoir_buffer.get_buffer_size() > 0 and task_id > 0:
                reservoir_images, reservoir_labels = reservoir_buffer.get_samples_from_buffer(
                    batch_size=current_batch_size, device=device)
                if reservoir_images is not None:
                    total_images = torch.cat([current_images, reservoir_images])
                    total_labels = torch.cat([current_labels, reservoir_labels])

            optimizer.zero_grad()
            outputs = model(total_images)
            loss = criterion(outputs, total_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == total_labels).sum().item()
            train_total += total_labels.size(0)
            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            tqdm_train_loader.set_description(f"Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}")
        
        scheduler.step()
        tqdm_epochs.set_description(f"Task {task_id+1} - Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%")


def run_experiment(dataset_name, buffer_size):
    """
    Runs a complete continual learning experiment and calculates advanced metrics.
    """
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    logger.info("=" * 70)
    logger.info(f"STARTING EXPERIMENT: Dataset={dataset_name}, Buffer Size={buffer_size}")
    logger.info(f"Epochs: {Hyperparameters.EPOCHS_PER_TASK}, LR: {Hyperparameters.LEARNING_RATE}, Batch: {Hyperparameters.BATCH_SIZE}")
    logger.info("=" * 70)

    # --- Data Loading ---
    if dataset_name == "split_cifar10":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    elif dataset_name == "split_cifar100":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    elif dataset_name == "split_mnist":
        train_loaders, test_loaders, num_classes, task_classes = load_data_split_mnist_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    num_tasks = len(train_loaders)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))

    # --- Model, Optimizer, Buffer Initialization ---
    model = ResNet18_CIFAR(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # --- Get Baseline Accuracies (for Forward Transfer) ---
    logger.info("Calculating baseline accuracies on a randomly initialized model...")
    criterion = nn.CrossEntropyLoss()
    baseline_accuracies = []
    for task_id in range(num_tasks):
        _, test_acc = evaluate_model(model, device, test_loaders[task_id], criterion)
        baseline_accuracies.append(test_acc)
    logger.info(f"Baseline Accuracies: {[f'{acc:.2f}%' for acc in baseline_accuracies]}")

    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY, nesterov=True)
    reservoir_buffer = ReservoirBuffer(buffer_size=buffer_size)
    
    # --- Training and Evaluation Loop ---
    for i, train_loader in enumerate(train_loaders):
        logger.info("-" * 60)
        logger.info(f"Training on Task {i+1}/{num_tasks} (Classes: {task_classes[i]})")
        logger.info("-" * 60)

        # Reset learning rate for each task
        for param_group in optimizer.param_groups:
            param_group['lr'] = Hyperparameters.LEARNING_RATE
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Hyperparameters.EPOCHS_PER_TASK, eta_min=1e-4)

        train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=i)
        
        # Evaluate on all tasks after training on task i
        for j in range(num_tasks):
            _, test_acc = evaluate_model(model, device, test_loaders[j], criterion)
            accuracy_matrix[i, j] = test_acc
        logger.info(f"Accuracies after task {i+1}: {[f'{acc:.2f}%' for acc in accuracy_matrix[i]]}")

        # --- Corrected Buffer Population ---
        # 1. Populate the buffer with samples from the current task's data.
        logger.info(f"Populating reservoir buffer with data from task {i+1}...")
        model.eval() # Set model to eval mode to avoid affecting batchnorm stats, etc.
        for images, labels, _ in tqdm(train_loader, desc="Buffer Population"):
            reservoir_buffer.add_to_buffer(images, labels)
        logger.info(f"Buffer size is now: {reservoir_buffer.get_buffer_size()}")

    # --- Final Metrics Calculation ---
    logger.info("=" * 60)
    logger.info("FINAL METRICS CALCULATION")
    logger.info("=" * 60)

    final_accuracies = accuracy_matrix[-1, :]
    avg_accuracy = np.mean(final_accuracies)

    # Backward Transfer (BWT)
    bwt_list = [accuracy_matrix[num_tasks-1, j] - accuracy_matrix[j, j] for j in range(num_tasks - 1)]
    avg_bwt = np.mean(bwt_list) if bwt_list else 0.0

    # Forgetting (F)
    forgetting_list = []
    for j in range(num_tasks - 1):
        max_acc = np.max(accuracy_matrix[j:, j])
        final_acc = accuracy_matrix[num_tasks - 1, j]
        forgetting_list.append(max_acc - final_acc)
    avg_forgetting = np.mean(forgetting_list) if forgetting_list else 0.0

    # Forward Transfer (FWT)
    fwt_list = [accuracy_matrix[i-1, i] - baseline_accuracies[i] for i in range(1, num_tasks)]
    avg_fwt = np.mean(fwt_list) if fwt_list else 0.0

    logger.info(f"Final Accuracy Matrix:\n{np.round(accuracy_matrix, 2)}")
    logger.info("-" * 60)
    logger.info(f"FINAL AVERAGE ACCURACY: {avg_accuracy:.2f}%")
    logger.info(f"AVERAGE FORGETTING (F): {avg_forgetting:.2f}%")
    logger.info(f"AVERAGE BACKWARD TRANSFER (BWT): {avg_bwt:.2f}%")
    logger.info(f"AVERAGE FORWARD TRANSFER (FWT): {avg_fwt:.2f}%")
    logger.info("=" * 70)
    
    return {
        "dataset": dataset_name,
        "buffer_size": buffer_size,
        "avg_accuracy": avg_accuracy,
        "avg_forgetting": avg_forgetting,
        "avg_bwt": avg_bwt,
        "avg_fwt": avg_fwt
    }

def print_results_table(results):
    """Prints a formatted table of experiment results."""
    logger.info("\n" + "=" * 125)
    logger.info(" " * 50 + "FINAL EXPERIMENT RESULTS")
    logger.info("=" * 125)
    
    # Header
    header = (f"| {'Dataset':<20} | {'Buffer Size':<12} | {'Avg. Accuracy (%)':<20} | "
              f"{'Forgetting (%)':<17} | {'BWT (%)':<15} | {'FWT (%)':<15} |")
    logger.info(header)
    logger.info(f"|{'-'*22}|{'-'*14}|{'-'*22}|{'-'*19}|{'-'*17}|{'-'*17}|")

    # Rows
    for result in results:
        row = (f"| {result['dataset']:<20} | {result['buffer_size']:<12} | "
               f"{result['avg_accuracy']:.2f}{'':<16} | "
               f"{result['avg_forgetting']:.2f}{'':<13} | "
               f"{result['avg_bwt']:.2f}{'':<11} | "
               f"{result['avg_fwt']:.2f}{'':<11} |")
        logger.info(row)
        
    logger.info("=" * 125)

# Main execution
if __name__ == '__main__':
    
    datasets_to_run = [ "split_cifar10", "split_cifar100"]
    buffer_sizes_to_run = [200, 500, 1000, 2000]
    
    all_results = []
    
    for dataset in datasets_to_run:
        for buffer_size in buffer_sizes_to_run:
            try:
                result = run_experiment(dataset_name=dataset, buffer_size=buffer_size)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed for {dataset} with buffer size {buffer_size}: {e}", exc_info=True)
                # Add a failed result to the table for completeness
                all_results.append({
                    "dataset": dataset, "buffer_size": buffer_size, "avg_accuracy": 0.0,
                    "avg_forgetting": 0.0, "avg_bwt": 0.0, "avg_fwt": 0.0
                })

    # Display final summary table
    print_results_table(all_results)
    
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    sys.exit(0)