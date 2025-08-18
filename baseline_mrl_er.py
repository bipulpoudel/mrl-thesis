import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from load_data_tasks import load_data_cifar10_by_tasks, load_data_cifar100_by_tasks
from logger import logger
from utils import optimize_device_for_pytorch, set_reproducibility_seeds
import torch.optim as optim
import random
import sys
import numpy as np

class Hyperparameters:
    EPOCHS_PER_TASK = 5 
    SEED = 42
    DATA_DIR = "./data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4 
    # Matryoshka Representation Learning Dimensions
    MATROSHKA_DIMS = [64, 128, 256, 512]
#====================================================================================
# MATRYOSHKA REPRESENTATION LEARNING IMPLEMENTATION
#====================================================================================

class MatryoshkaResNet18(nn.Module):
    def __init__(self, num_classes, matryoshka_dims, pretrained=False):
        super().__init__()
        # 1. Use the same pre-trained weight logic as the baseline
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
            
        # 2. Get the original feature dimension BEFORE modifying the final layer
        feature_dim = self.backbone.fc.in_features
        
        # 3. Apply the EXACT same modifications for CIFAR as the baseline model
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity() # Remove max pooling for small images
        
        # 4. Turn the backbone into a feature extractor
        self.backbone.fc = nn.Identity()

        # 5. Initialize Matryoshka heads (this logic remains the same)
        self.matryoshka_dims = sorted(matryoshka_dims)
        if self.matryoshka_dims[-1] != feature_dim:
            raise ValueError(f"The largest Matryoshka dimension must match the backbone's feature dimension of {feature_dim}")
            
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_classes) for dim in self.matryoshka_dims
        ])

    def forward(self, x):
        features = self.backbone(x)
        outputs = []
        for i, dim in enumerate(self.matryoshka_dims):
            nested_features = features[:, :dim]
            output = self.heads[i](nested_features)
            outputs.append(output)
        return outputs

## CORRECTED ##: Use a simple, unweighted sum for the MRL loss.
def matryoshka_loss(matryoshka_outputs, labels, criterion):
    """
    Calculates the standard Matryoshka loss by summing the loss from each head.
    This ensures that the initial dimensions of the feature representation receive
    the strongest cumulative gradient signal.
    """
    total_loss = 0
    for output in matryoshka_outputs:
        total_loss += criterion(output, labels)
    return total_loss

#====================================================================================

class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.n_seen_so_far = 0

    def add_to_buffer(self, items, labels):
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
    num_heads = len(Hyperparameters.MATROSHKA_DIMS)
    val_losses = [0.0] * num_heads
    val_corrects = [0] * num_heads
    val_total = 0
    with torch.no_grad():
        for i, (images, labels, task_id) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            matryoshka_outputs = model(images)
            val_total += labels.size(0)
            for head_idx, outputs in enumerate(matryoshka_outputs):
                loss = criterion(outputs, labels)   
                val_losses[head_idx] += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                val_corrects[head_idx] += (predicted == labels).sum().item()

    avg_losses = [loss / len(validation_loader) for loss in val_losses]
    accuracies = [100 * correct / val_total for correct in val_corrects]
    return avg_losses, accuracies

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
                    # This doubles effective batch size, which is a valid strategy.
                    total_images = torch.cat([current_images, reservoir_images])
                    total_labels = torch.cat([current_labels, reservoir_labels])

            optimizer.zero_grad()
            
            # ## MRL CHANGE ##: MRL forward pass and loss calculation
            matryoshka_outputs = model(total_images)
             ## MODIFIED ##: Pass the weights to the loss function.
            loss = matryoshka_loss(
                matryoshka_outputs, 
                total_labels, 
                criterion
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            # ## MRL CHANGE ##: Accuracy is calculated using the largest head for stable reporting
            full_dim_output = matryoshka_outputs[-1]
            train_correct += (full_dim_output.argmax(dim=1) == total_labels).sum().item()
            train_total += total_labels.size(0)

            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            tqdm_train_loader.set_description(f"Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}")
        
        scheduler.step()
        tqdm_epochs.set_description(f"Task {task_id+1} - Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%")

def run_experiment(dataset_name, buffer_size):
    """
    Runs a complete continual learning experiment with the MRL model.
    """
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    logger.info("=" * 70)
    logger.info(f"STARTING MRL EXPERIMENT: Dataset={dataset_name}, Buffer Size={buffer_size}")
    logger.info(f"Epochs: {Hyperparameters.EPOCHS_PER_TASK}, LR: {Hyperparameters.LEARNING_RATE}, Batch: {Hyperparameters.BATCH_SIZE}")
    logger.info("=" * 70)

    # --- Data Loading ---
    if dataset_name == "split_cifar10":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    elif dataset_name == "split_cifar100":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    num_tasks = len(train_loaders)
    num_heads = len(Hyperparameters.MATROSHKA_DIMS)

    # ==============================================================================
    # == CRITICAL FWT FIX & MODEL INITIALIZATION ===================================
    # ==============================================================================
    # Initialize the model ONCE with pre-trained weights
    model = MatryoshkaResNet18(
        num_classes=num_classes, 
        matryoshka_dims=Hyperparameters.MATROSHKA_DIMS, 
        pretrained=True  # Use pre-trained for both baseline and experiment
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # --- Get Baseline Accuracies (for Forward Transfer) ---
    logger.info("Calculating baseline accuracies on the initial pre-trained model...")
    initial_state_dict = model.state_dict() # Save initial state
    
    baseline_accuracies = []
    for task_id in range(num_tasks):
        _, test_accs = evaluate_model(model, device, test_loaders[task_id], criterion)
        baseline_accuracies.append(test_accs[-1]) # Use largest head for baseline
    logger.info(f"Baseline Accuracies (Full Dim): {[f'{acc:.2f}%' for acc in baseline_accuracies]}")

    # Restore the initial state to ensure the experiment starts from a clean slate
    model.load_state_dict(initial_state_dict)
    
    # --- Optimizer, Scheduler, and Buffer Setup ---
    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY, nesterov=True)
    reservoir_buffer = ReservoirBuffer(buffer_size=buffer_size)

    # ==============================================================================
    # ## MODIFICATION ##: Create a list of accuracy matrices, one for each head
    # ==============================================================================
    accuracy_matrices = [np.zeros((num_tasks, num_tasks)) for _ in range(num_heads)]
    
    # --- Training and Evaluation Loop ---
    for i, train_loader in enumerate(train_loaders):
        logger.info("-" * 60)
        logger.info(f"Processing Task {i+1}/{num_tasks} (Classes: {task_classes[i]})")

        # Reset learning rate for each task
        for param_group in optimizer.param_groups:
            param_group['lr'] = Hyperparameters.LEARNING_RATE
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Hyperparameters.EPOCHS_PER_TASK, # T_max is now just the epochs for THIS task
            eta_min=1e-5 # Can use a slightly larger eta_min if desired
        )
        
        train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=i)
        
        # Evaluate on all tasks after training on task i
        for j in range(num_tasks):
            _, test_accs = evaluate_model(model, device, test_loaders[j], criterion)
            
            # ## MODIFICATION ##: Populate the matrix for each head
            for head_idx, acc in enumerate(test_accs):
                accuracy_matrices[head_idx][i, j] = acc
            
            # Log accuracies of all heads for detailed analysis
            if j <= i:
                acc_log = ", ".join([f"Dim {d}: {a:.2f}%" for d, a in zip(Hyperparameters.MATROSHKA_DIMS, test_accs)])
                logger.info(f"  Eval on Task {j+1}: {acc_log}")

        # Populate buffer after training and evaluation
        logger.info(f"Populating reservoir buffer with data from task {i+1}...")
        for images, labels, _ in tqdm(train_loader, desc="Buffer Population"):
            reservoir_buffer.add_to_buffer(images, labels)
        logger.info(f"Buffer size is now: {reservoir_buffer.get_buffer_size()}")

        # Log the performance of the largest head after each task
        logger.info(f"Accuracies after task {i+1} (Full Dim): {[f'{acc:.2f}%' for acc in accuracy_matrices[-1][i]]}")

    # ==============================================================================
    # == FINAL METRICS CALCULATION =================================================
    # ==============================================================================
    logger.info("=" * 60)
    logger.info("FINAL METRICS CALCULATION")
    logger.info("=" * 60)

    # --- Standard CL metrics are calculated on the LARGEST head ---
    accuracy_matrix_full_dim = accuracy_matrices[-1]
    final_accuracies = accuracy_matrix_full_dim[-1, :]
    avg_accuracy = np.mean(final_accuracies)
    bwt_list = [accuracy_matrix_full_dim[num_tasks-1, j] - accuracy_matrix_full_dim[j, j] for j in range(num_tasks - 1)]
    avg_bwt = np.mean(bwt_list) if bwt_list else 0.0
    forgetting_list = []
    for j in range(num_tasks - 1):
        max_acc = np.max(accuracy_matrix_full_dim[j:, j])
        final_acc = accuracy_matrix_full_dim[num_tasks - 1, j]
        forgetting_list.append(max_acc - final_acc)
    avg_forgetting = np.mean(forgetting_list) if forgetting_list else 0.0
    fwt_list = [accuracy_matrix_full_dim[i-1, i] - baseline_accuracies[i] for i in range(1, num_tasks)]
    avg_fwt = np.mean(fwt_list) if fwt_list else 0.0

    logger.info(f"Final Accuracy Matrix (Full Dim):\n{np.round(accuracy_matrix_full_dim, 2)}")
    logger.info("-" * 60)
    logger.info("--- Metrics for Full-Dimension Head (Dim 512) ---")
    logger.info(f"FINAL AVERAGE ACCURACY: {avg_accuracy:.2f}%")
    logger.info(f"AVERAGE FORGETTING (F): {avg_forgetting:.2f}%")
    logger.info(f"AVERAGE BACKWARD TRANSFER (BWT): {avg_bwt:.2f}%")
    logger.info(f"AVERAGE FORWARD TRANSFER (FWT): {avg_fwt:.2f}%")
    logger.info("-" * 60)
    
    final_avg_acc_per_head = {}
    logger.info("--- Final Average Accuracy Per Matryoshka Head ---")
    for head_idx, dim in enumerate(Hyperparameters.MATROSHKA_DIMS):
        # Get the final row of accuracies for this head (performance on all tasks after all training)
        final_accuracies_head = accuracy_matrices[head_idx][-1, :]
        avg_acc_head = np.mean(final_accuracies_head)
        final_avg_acc_per_head[dim] = avg_acc_head
        logger.info(f"  FINAL AVG ACCURACY (Dim {dim}): {avg_acc_head:.2f}%")
    
    # The main dictionary returned is still based on the largest head for consistency
    # in the final results table. The detailed breakdown is in the logs.
    return {
        "dataset": dataset_name, "buffer_size": buffer_size, "avg_accuracy": avg_accuracy,
        "avg_forgetting": avg_forgetting, "avg_bwt": avg_bwt, "avg_fwt": avg_fwt,
        "final_accuracies_per_head": final_avg_acc_per_head # Optional: return for more analysis
    }

def print_results_table(results):
    logger.info("\n" + "=" * 125)
    logger.info(" " * 50 + "FINAL EXPERIMENT RESULTS")
    logger.info("=" * 125)
    header = (f"| {'Dataset':<20} | {'Buffer Size':<12} | {'Avg. Accuracy (%)':<20} | "
              f"{'Forgetting (%)':<17} | {'BWT (%)':<15} | {'FWT (%)':<15} |")
    logger.info(header)
    logger.info(f"|{'-'*22}|{'-'*14}|{'-'*22}|{'-'*19}|{'-'*17}|{'-'*17}|")
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
    datasets_to_run = ["split_cifar10", "split_cifar100"]
    buffer_sizes_to_run = [200, 500, 1000, 2000]
    
    all_results = []
    
    for dataset in datasets_to_run:
        for buffer_size in buffer_sizes_to_run:
            try:
                result = run_experiment(dataset_name=dataset, buffer_size=buffer_size)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed for {dataset} with buffer size {buffer_size}: {e}", exc_info=True)
                all_results.append({
                    "dataset": dataset, "buffer_size": buffer_size, "avg_accuracy": 0.0,
                    "avg_forgetting": 0.0, "avg_bwt": 0.0, "avg_fwt": 0.0
                })

    print_results_table(all_results)
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    sys.exit(0)