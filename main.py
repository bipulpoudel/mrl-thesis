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

class Hyperparameters:
    # Increased epochs for better convergence
    EPOCHS_PER_TASK = 5 
    
    #Seed
    SEED = 42
    
    #Data directory
    DATA_DIR = "./data"
    
    #Batch sizes and number of workers
    BATCH_SIZE = 32  # Reduced from 128 for better gradient estimation
    NUM_WORKERS = 4
    
    #Number of classes
    NUM_CLASSES = 10
    
    #Learning rate - increased for better convergence
    LEARNING_RATE = 0.01  # Increased from 0.001
    
    #Momentum
    MOMENTUM = 0.9
    
    #Weight decay
    WEIGHT_DECAY = 5e-4 
    
    #Buffer size
    BUFFER_SIZE = 2000

#Improved Reservoir Buffer with better sampling strategy
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
                # Reservoir sampling
                j = random.randint(0, self.n_seen_so_far)
                if j < self.buffer_size:
                    self.buffer[j] = sample
            
            self.n_seen_so_far += 1

    def get_buffer_size(self):
        return len(self.buffer)
    
    def get_samples_from_buffer(self, batch_size, device):
        """
        Get random samples from buffer for replay.
        """
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
    """
    Improved training function with proper replay buffer management
    """
    tqdm_epochs = tqdm(range(epochs), desc=f"Training Task {task_id+1}")
    
    for epoch in tqdm_epochs:
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        tqdm_train_loader = tqdm(train_loader, desc="Progress:", leave=False)
        
        for i, (images, labels, task_id_batch) in enumerate(tqdm_train_loader):
            current_images, current_labels = images.to(device), labels.to(device)
            
            # Determine batch composition
            current_batch_size = current_images.size(0)
            
            # During training, add current samples to buffer with reservoir sampling
            if epoch == 0:  # Only add samples once per task
                reservoir_buffer.add_to_buffer(current_images, current_labels)
            
            # Get replay samples if buffer has data
            buffer_size = reservoir_buffer.get_buffer_size()
            if buffer_size > 0 and task_id > 0:  # Only replay for tasks after the first
                
                # Get replay samples
                reservoir_images, reservoir_labels = reservoir_buffer.get_samples_from_buffer(
                    batch_size=current_batch_size, device=device)
                
                # Combine current and replay samples
                if reservoir_images is not None:
                    total_images = torch.cat([current_images, reservoir_images])
                    total_labels = torch.cat([current_labels, reservoir_labels])
                else:
                    total_images = current_images
                    total_labels = current_labels
            else:
                total_images = current_images
                total_labels = current_labels

            optimizer.zero_grad()
            outputs = model(total_images)
            loss = criterion(outputs, total_labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == total_labels).sum().item()
            train_total += total_labels.size(0)

            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            
            tqdm_train_loader.set_description(
                f"Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}"
            )
        
        # Update learning rate
        scheduler.step()
        
        # Update epoch progress bar
        tqdm_epochs.set_description(
            f"Task {task_id+1} - Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%"
        )

def cosine_annealing_with_restarts(optimizer, T_max, T_mult=1, eta_min=0, last_epoch=-1):
    """
    Cosine annealing with warm restarts scheduler
    """
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_max, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
    )

# Main execution
if __name__ == '__main__':
    
    dataset = "split_cifar10"
    num_classes = 10

    set_reproducibility_seeds(Hyperparameters.SEED)
    
    device = optimize_device_for_pytorch()
    
    logger.info("=" * 60)
    logger.info(f"IMPROVED ER PIPELINE FOR {dataset}")
    logger.info(f"Buffer Size: {Hyperparameters.BUFFER_SIZE}")
    logger.info(f"Epochs per Task: {Hyperparameters.EPOCHS_PER_TASK}")
    logger.info(f"Learning Rate: {Hyperparameters.LEARNING_RATE}")
    logger.info(f"Batch Size: {Hyperparameters.BATCH_SIZE}")
    logger.info("=" * 60)

    if dataset == "split_cifar10":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks(
            batch_size=Hyperparameters.BATCH_SIZE,
        )
        num_classes = 10
    elif dataset == "split_cifar100":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks(
            batch_size=Hyperparameters.BATCH_SIZE
        )
        num_classes = 100
    elif dataset == "split_mnist":
        train_loaders, test_loaders, num_classes, task_classes = load_data_split_mnist_by_tasks(
            batch_size=Hyperparameters.BATCH_SIZE
        )
        num_classes = 10
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # Initialize model
    model = ResNet18_CIFAR(num_classes=num_classes, pretrained=True)  # Don't use pretrained for CL
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    model.to(device)

    # Initialize optimizer with higher learning rate
    optimizer = optim.SGD(
        model.parameters(), 
        lr=Hyperparameters.LEARNING_RATE, 
        momentum=Hyperparameters.MOMENTUM, 
        weight_decay=Hyperparameters.WEIGHT_DECAY,
        nesterov=True  # Add Nesterov momentum
    )
    
    # Use cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=Hyperparameters.EPOCHS_PER_TASK * len(train_loaders),
        eta_min=1e-4
    )

    # Initialize reservoir buffer
    reservoir_buffer = ReservoirBuffer(buffer_size=Hyperparameters.BUFFER_SIZE)

    # Track accuracies
    all_task_accuracies = []
    
    for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        logger.info("=" * 60)
        logger.info(f"Training on Task {i+1} of {len(train_loaders)}")
        logger.info(f"Classes in this task: {task_classes[i]}")
        logger.info("=" * 60)
        
        criterion = nn.CrossEntropyLoss()
        
        # Train model with improved function
        train_model(
            model, device, train_loader, criterion, 
            reservoir_buffer, optimizer, scheduler, 
            epochs=Hyperparameters.EPOCHS_PER_TASK, 
            task_id=i
        )
        
        # Evaluate on all tasks seen so far
        task_accuracies = []
        average_test_acc = 0.0
        
        for j in range(i + 1):
            test_loader_current = test_loaders[j]
            test_loss, test_acc = evaluate_model(model, device, test_loader_current, criterion)
            
            logger.info(f"Task {j+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
            task_accuracies.append(test_acc)
            average_test_acc += test_acc
        
        average_test_acc = average_test_acc / (i + 1)
        logger.info(f"Average Accuracy (Tasks 1-{i+1}): {average_test_acc:.2f}%")
        
        all_task_accuracies.append(task_accuracies)
    
    # Final evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON ALL TASKS")
    logger.info("=" * 60)
    
    final_accuracies = []
    for j, test_loader in enumerate(test_loaders):
        test_loss, test_acc = evaluate_model(model, device, test_loader, criterion)
        final_accuracies.append(test_acc)
        logger.info(f"Task {j+1} - Final Accuracy: {test_acc:.2f}%")
    
    final_average = sum(final_accuracies) / len(final_accuracies)
    logger.info("=" * 60)
    logger.info(f"FINAL AVERAGE ACCURACY: {final_average:.2f}%")
    logger.info("=" * 60)
    
    # Calculate forgetting
    if len(all_task_accuracies) > 1:
        forgetting = []
        for task_id in range(len(all_task_accuracies) - 1):
            max_acc = max([accs[task_id] for accs in all_task_accuracies[task_id:]])
            final_acc = final_accuracies[task_id]
            task_forgetting = max_acc - final_acc
            forgetting.append(task_forgetting)
            logger.info(f"Task {task_id+1} - Forgetting: {task_forgetting:.2f}%")
        
        avg_forgetting = sum(forgetting) / len(forgetting)
        logger.info(f"Average Forgetting: {avg_forgetting:.2f}%")
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    sys.exit(0) 