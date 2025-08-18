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
# SIMPLIFIED MRL - FOCUS ON FULL DIMENSION ONLY
#====================================================================================

class MatryoshkaResNet18(nn.Module):
    def __init__(self, num_classes, matryoshka_dims, pretrained=False):
        super().__init__()
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
            
        feature_dim = self.backbone.fc.in_features
        
        # Apply the same modifications for CIFAR as the baseline model
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        # Turn the backbone into a feature extractor
        self.backbone.fc = nn.Identity()

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

def matryoshka_loss_full_only(matryoshka_outputs, labels, criterion):
    """
    SIMPLIFIED: Only compute loss for the full-dimension head.
    This makes MRL behave like the baseline model.
    """
    # Only use the last (full-dimension) output
    return criterion(matryoshka_outputs[-1], labels)

def matryoshka_loss_weighted_simple(matryoshka_outputs, labels, criterion):
    """
    ALTERNATIVE: Heavily weight the full dimension, minimal weight to others.
    This maintains some MRL structure but focuses on full dimension performance.
    """
    total_loss = 0
    num_heads = len(matryoshka_outputs)
    
    # Give 90% weight to full dimension, 10% distributed to others
    for i, output in enumerate(matryoshka_outputs):
        if i == num_heads - 1:  # Full dimension
            weight = 0.9
        else:
            weight = 0.1 / (num_heads - 1)  # Split remaining 10% among other heads
        total_loss += weight * criterion(output, labels)
    
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
                epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=0, loss_strategy="full_only"):
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
            
            # Choose loss strategy
            matryoshka_outputs = model(total_images)
            if loss_strategy == "full_only":
                loss = matryoshka_loss_full_only(matryoshka_outputs, total_labels, criterion)
            elif loss_strategy == "weighted_simple":
                loss = matryoshka_loss_weighted_simple(matryoshka_outputs, total_labels, criterion)
            else:  # original
                total_loss = 0
                for output in matryoshka_outputs:
                    total_loss += criterion(output, total_labels)
                loss = total_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            # Accuracy using the largest head
            full_dim_output = matryoshka_outputs[-1]
            train_correct += (full_dim_output.argmax(dim=1) == total_labels).sum().item()
            train_total += total_labels.size(0)

            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            tqdm_train_loader.set_description(f"Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}")
        
        scheduler.step()
        tqdm_epochs.set_description(f"Task {task_id+1} - Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%")

def run_experiment(dataset_name, buffer_size, loss_strategy="full_only"):
    """
    Runs experiment with simplified MRL that should match baseline performance.
    
    Args:
        dataset_name: Name of the dataset
        buffer_size: Size of the replay buffer
        loss_strategy: "full_only" (only train full dim), "weighted_simple" (90% full, 10% others), or "original"
    """
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    logger.info("=" * 70)
    logger.info(f"STARTING SIMPLIFIED MRL EXPERIMENT")
    logger.info(f"Dataset={dataset_name}, Buffer Size={buffer_size}, Strategy={loss_strategy}")
    logger.info(f"Epochs: {Hyperparameters.EPOCHS_PER_TASK}, LR: {Hyperparameters.LEARNING_RATE}")
    logger.info("=" * 70)

    # Data Loading
    if dataset_name == "split_cifar10":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    elif dataset_name == "split_cifar100":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    num_tasks = len(train_loaders)
    num_heads = len(Hyperparameters.MATROSHKA_DIMS)

    # Initialize model with pre-trained weights (same as baseline)
    model = MatryoshkaResNet18(
        num_classes=num_classes, 
        matryoshka_dims=Hyperparameters.MATROSHKA_DIMS, 
        pretrained=True
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Get baseline accuracies
    logger.info("Calculating baseline accuracies...")
    initial_state_dict = model.state_dict()
    
    baseline_accuracies = []
    for task_id in range(num_tasks):
        _, test_accs = evaluate_model(model, device, test_loaders[task_id], criterion)
        baseline_accuracies.append(test_accs[-1])
    logger.info(f"Baseline Accuracies: {[f'{acc:.2f}%' for acc in baseline_accuracies]}")

    # Restore initial state
    model.load_state_dict(initial_state_dict)
    
    # Setup optimizer and buffer
    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, 
                         momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY, nesterov=True)
    reservoir_buffer = ReservoirBuffer(buffer_size=buffer_size)

    accuracy_matrices = [np.zeros((num_tasks, num_tasks)) for _ in range(num_heads)]
    
    # Training loop
    for i, train_loader in enumerate(train_loaders):
        logger.info("-" * 60)
        logger.info(f"Processing Task {i+1}/{num_tasks} (Classes: {task_classes[i]})")

        # Reset learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = Hyperparameters.LEARNING_RATE
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Hyperparameters.EPOCHS_PER_TASK, eta_min=1e-5
        )
        
        train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, 
                   epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=i, loss_strategy=loss_strategy)
        
        # Evaluate on all tasks
        for j in range(num_tasks):
            _, test_accs = evaluate_model(model, device, test_loaders[j], criterion)
            for head_idx, acc in enumerate(test_accs):
                accuracy_matrices[head_idx][i, j] = acc
            
            if j <= i:
                logger.info(f"  Task {j+1} - Full Dim Acc: {test_accs[-1]:.2f}%")

        # Populate buffer
        logger.info(f"Populating buffer with task {i+1} data...")
        for images, labels, _ in tqdm(train_loader, desc="Buffer", leave=False):
            reservoir_buffer.add_to_buffer(images, labels)
        logger.info(f"Buffer size: {reservoir_buffer.get_buffer_size()}")

    # Calculate final metrics (using full dimension head)
    logger.info("=" * 60)
    logger.info("FINAL METRICS")
    logger.info("=" * 60)

    accuracy_matrix_full = accuracy_matrices[-1]
    final_accuracies = accuracy_matrix_full[-1, :]
    avg_accuracy = np.mean(final_accuracies)
    
    # Calculate other metrics
    bwt_list = [accuracy_matrix_full[num_tasks-1, j] - accuracy_matrix_full[j, j] for j in range(num_tasks - 1)]
    avg_bwt = np.mean(bwt_list) if bwt_list else 0.0
    
    forgetting_list = []
    for j in range(num_tasks - 1):
        max_acc = np.max(accuracy_matrix_full[j:, j])
        final_acc = accuracy_matrix_full[num_tasks - 1, j]
        forgetting_list.append(max_acc - final_acc)
    avg_forgetting = np.mean(forgetting_list) if forgetting_list else 0.0
    
    fwt_list = [accuracy_matrix_full[i-1, i] - baseline_accuracies[i] for i in range(1, num_tasks)]
    avg_fwt = np.mean(fwt_list) if fwt_list else 0.0

    logger.info(f"Strategy: {loss_strategy}")
    logger.info(f"FINAL AVERAGE ACCURACY: {avg_accuracy:.2f}%")
    logger.info(f"AVERAGE FORGETTING: {avg_forgetting:.2f}%")
    logger.info(f"AVERAGE BWT: {avg_bwt:.2f}%")
    logger.info(f"AVERAGE FWT: {avg_fwt:.2f}%")
    
    # Show performance of other heads
    logger.info("\n--- Performance of Other Heads ---")
    for head_idx, dim in enumerate(Hyperparameters.MATROSHKA_DIMS):
        final_acc_head = np.mean(accuracy_matrices[head_idx][-1, :])
        logger.info(f"  Dim {dim}: {final_acc_head:.2f}%")
    
    return {
        "dataset": dataset_name, "buffer_size": buffer_size, 
        "avg_accuracy": avg_accuracy, "avg_forgetting": avg_forgetting,
        "avg_bwt": avg_bwt, "avg_fwt": avg_fwt, "strategy": loss_strategy
    }

if __name__ == '__main__':
    # Test different loss strategies
    strategies = ["full_only"]
    dataset = "split_cifar10"
    buffer_size = 200
    
    results = []
    for strategy in strategies:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing strategy: {strategy}")
        logger.info(f"{'='*70}")
        
        result = run_experiment(dataset, buffer_size, loss_strategy=strategy)
        results.append(result)
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("COMPARISON OF STRATEGIES")
    logger.info("="*70)
    logger.info(f"{'Strategy':<20} | {'Avg Accuracy':<15} | {'Forgetting':<15}")
    logger.info("-"*70)
    for r in results:
        logger.info(f"{r['strategy']:<20} | {r['avg_accuracy']:.2f}%{'':<10} | {r['avg_forgetting']:.2f}%")
    
    logger.info("\nExperiment complete!")
    sys.exit(0)
