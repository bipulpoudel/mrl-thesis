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
import torch.nn.functional as F
import copy # Used for deep copying the model's state
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import json # Added for JSON output
import os   # Added for directory creation

class Hyperparameters:
    # Maximum epochs per task, may be cut short by early stopping
    EPOCHS_PER_TASK = 10 
    
    #Seed
    SEED = 42
    
    #Data directory
    DATA_DIR = "./data"
    
    #Batch sizes and number of workers
    BATCH_SIZE = 32
    
    #Learning rate
    LEARNING_RATE = 0.01
    
    #Momentum
    MOMENTUM = 0.9
    
    #Weight decay
    WEIGHT_DECAY = 5e-4 

    # --- Knowledge Distillation Parameters ---
    KD_ALPHA = 0.5
    KD_TEMPERATURE = 2.0

    # --- Early Stopping Parameters ---
    # If validation accuracy doesn't improve for this many consecutive epochs, stop training for the current task.
    EARLY_STOPPING_PATIENCE = 10


# --- ReservoirBuffer (Unchanged) ---
class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.n_seen_so_far = 0

    def add_to_buffer(self, items, labels, task_ids):
        task_id = task_ids[0].item() 
        for item, label in zip(items, labels):
            sample = (item.detach().cpu(), label.detach().cpu(), task_id)
            if self.n_seen_so_far < self.buffer_size:
                self.buffer.append(sample)
            else:
                j = random.randint(0, self.n_seen_so_far)
                if j < self.buffer_size:
                    self.buffer[j] = sample
            self.n_seen_so_far += 1

    def get_buffer_size(self):
        return len(self.buffer)
    
    def get_samples_from_buffer(self, batch_size, device, current_task_id):
        past_samples = [s for s in self.buffer if s[2] != current_task_id]
        if not past_samples:
            return None, None
        num_samples_to_get = min(len(past_samples), batch_size)
        random_samples = random.sample(past_samples, num_samples_to_get)
        items, labels, _ = zip(*random_samples)
        items_tensor = torch.stack(items).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        return items_tensor, labels_tensor

# --- Knowledge Distillation Loss (Unchanged) ---
def distillation_loss(student_logits, teacher_logits, temperature):
    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student_probs = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(log_soft_student_probs, soft_teacher_probs.detach(), reduction='batchmean')
    return loss * (temperature ** 2)


# --- evaluate_model (Unchanged) ---
def evaluate_model(model, device, validation_loader, criterion):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _ in validation_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            outputs = model(images)
            loss = criterion(outputs, labels)   
            val_loss += loss.item() * batch_size 
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += batch_size
    avg_loss = val_loss / val_total if val_total else 0.0
    avg_acc  = 100 * val_correct / val_total if val_total else 0.0
    return avg_loss, avg_acc

# --- train_model (Unchanged from previous modification) ---
def train_model(model, device, train_loader, validation_loader, criterion, reservoir_buffer, optimizer, scheduler, 
                epochs=10, task_id=0, teacher_model=None, kd_mode='none'):
    
    best_val_acc = -1.0
    best_model_state = None
    epochs_no_improve = 0
    final_epoch = epochs

    tqdm_epochs = tqdm(range(epochs), desc=f"Training Task {task_id+1} (KD: {kd_mode})")
    for epoch in tqdm_epochs:
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']
        tqdm_train_loader = tqdm(train_loader, desc="Progress:", leave=False)
        
        for i, (images, labels, task_id_batch) in enumerate(tqdm_train_loader):
            current_images, current_labels = images.to(device), labels.to(device)
            current_batch_size = current_images.size(0)
            
            reservoir_images, reservoir_labels = None, None
            if task_id > 0:
                reservoir_images, reservoir_labels = reservoir_buffer.get_samples_from_buffer(
                    batch_size=current_batch_size, device=device, current_task_id=task_id)

            if reservoir_images is not None:
                total_images = torch.cat([current_images, reservoir_images])
                total_labels = torch.cat([current_labels, reservoir_labels])
            else:
                total_images = current_images
                total_labels = current_labels
            
            optimizer.zero_grad()
            outputs = model(total_images)
            
            loss_ce = criterion(outputs, total_labels)
            loss_kd = 0.0
            if teacher_model is not None and task_id > 0:
                with torch.no_grad():
                    teacher_logits_all = teacher_model(total_images)
                student_logits = outputs
                if kd_mode == 'all':
                    loss_kd = distillation_loss(student_logits, teacher_logits_all, Hyperparameters.KD_TEMPERATURE)

            loss = Hyperparameters.KD_ALPHA * loss_ce + (1 - Hyperparameters.KD_ALPHA) * loss_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            reservoir_buffer.add_to_buffer(images, labels, task_id_batch)

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == total_labels).sum().item()
            train_total += total_labels.size(0)
            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            tqdm_train_loader.set_description(f"Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}")
        
        avg_train_acc = 100 * train_correct / train_total if train_total else 0.0
        
        if validation_loader:
            val_loss, val_acc = evaluate_model(model, device, validation_loader, criterion)
            tqdm_epochs.set_description(f"Task {task_id+1} | Ep {epoch+1}/{epochs} | Tr Acc: {avg_train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                logger.info(f"  -> New best validation accuracy for task {task_id+1}: {best_val_acc:.2f}%")
            else:
                epochs_no_improve += 1
                logger.info(f"  -> No improvement in validation accuracy for {epochs_no_improve} epoch(s). Patience: {Hyperparameters.EARLY_STOPPING_PATIENCE}")
            
            if epochs_no_improve >= Hyperparameters.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered for task {task_id+1} at epoch {epoch + 1} after {epochs_no_improve} epochs without improvement.")
                final_epoch = epoch + 1
                break
        else:
            tqdm_epochs.set_description(f"Task {task_id+1} | Ep {epoch+1}/{epochs} | Tr Acc: {avg_train_acc:.2f}%")

        scheduler.step()

    if validation_loader and best_model_state:
        logger.info(f"Loading best model state for task {task_id+1} (Val Acc: {best_val_acc:.2f}%)")
        model.load_state_dict(best_model_state)
    elif validation_loader:
        logger.warning(f"No best model state was saved for task {task_id+1}; using the model from the last epoch.")

    return final_epoch

# --- Helper Class (Unchanged) ---
class ListAsDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- run_experiment (Unchanged from previous modification) ---
def run_experiment(dataset_name, buffer_size, kd_mode='none', epochs_per_task=10):
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    use_early_stopping = (epochs_per_task == 100)
    stop_epochs_list = []

    logger.info("=" * 70)
    logger.info(f"STARTING EXPERIMENT: Dataset={dataset_name}, Buffer={buffer_size}, KD={kd_mode}, Epochs={epochs_per_task}, EarlyStop={use_early_stopping}")
    logger.info("=" * 70)

    if dataset_name == "split_cifar10":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    elif dataset_name == "split_cifar100":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    num_tasks = len(train_loaders)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    model = ResNet18_CIFAR(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    
    baseline_accuracies = []
    for task_id in range(num_tasks):
        _, test_acc = evaluate_model(model, device, test_loaders[task_id], criterion)
        baseline_accuracies.append(test_acc)
    
    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY, nesterov=True)
    reservoir_buffer = ReservoirBuffer(buffer_size=buffer_size)
    teacher_model = None
    
    for i, train_loader_original in enumerate(train_loaders):
        logger.info("-" * 60)
        logger.info(f"Training on Task {i+1}/{num_tasks} (Classes: {task_classes[i]})")
        logger.info("-" * 60)

        if use_early_stopping:
            current_task_dataset = train_loader_original.dataset
            val_size = int(0.1 * len(current_task_dataset))
            train_size = len(current_task_dataset) - val_size
            train_subset, val_subset_current = random_split(
                current_task_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(Hyperparameters.SEED)
            )
            
            replay_samples = reservoir_buffer.buffer
            datasets_for_validation = [val_subset_current]
            if replay_samples:
                datasets_for_validation.append(ListAsDataset(replay_samples))
            
            validation_dataset = ConcatDataset(datasets_for_validation)
            train_loader_task = DataLoader(train_subset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=False)
            logger.info(f"Created training set for task {i+1} with {len(train_subset)} samples.")
            logger.info(f"Created validation set with {len(val_subset_current)} current + {len(replay_samples)} replay = {len(validation_dataset)} total samples.")
        else:
            train_loader_task = train_loader_original
            validation_loader = None
            logger.info(f"Training on full dataset for task {i+1} with {len(train_loader_task.dataset)} samples. Early stopping is OFF.")

        if i > 0 and kd_mode != 'none':
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()

        for param_group in optimizer.param_groups:
            param_group['lr'] = Hyperparameters.LEARNING_RATE
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_task, eta_min=1e-4)

        stop_epoch = train_model(model, device, train_loader_task, validation_loader, criterion, reservoir_buffer, optimizer, scheduler,
                                 epochs=epochs_per_task, task_id=i, teacher_model=teacher_model, kd_mode=kd_mode)
        stop_epochs_list.append(stop_epoch)
        
        for j in range(num_tasks):
            _, test_acc = evaluate_model(model, device, test_loaders[j], criterion)
            accuracy_matrix[i, j] = test_acc
        logger.info(f"Accuracies after task {i+1}: {[f'{acc:.2f}%' for acc in accuracy_matrix[i]]}")

    logger.info("=" * 60)
    logger.info(f"FINAL METRICS for KD Mode: {kd_mode}, Epochs: {epochs_per_task}")
    logger.info("=" * 60)

    final_accuracies = accuracy_matrix[-1, :]
    avg_accuracy = np.mean(final_accuracies)
    bwt_list = [accuracy_matrix[num_tasks-1, j] - accuracy_matrix[j, j] for j in range(num_tasks - 1)]
    avg_bwt = np.mean(bwt_list) if bwt_list else 0.0
    forgetting_list = [np.max(accuracy_matrix[j:, j]) - accuracy_matrix[num_tasks - 1, j] for j in range(num_tasks - 1)]
    avg_forgetting = np.mean(forgetting_list) if forgetting_list else 0.0
    fwt_list = [accuracy_matrix[i-1, i] - baseline_accuracies[i] for i in range(1, num_tasks)]
    avg_fwt = np.mean(fwt_list) if fwt_list else 0.0

    logger.info(f"Final Accuracy Matrix:\n{np.round(accuracy_matrix, 2)}")
    logger.info("-" * 60)
    logger.info(f"FINAL AVERAGE ACCURACY: {avg_accuracy:.2f}%")
    
    return {
        "dataset": dataset_name, "buffer_size": buffer_size, "kd_mode": kd_mode,
        "epochs_per_task": epochs_per_task,
        "early_stopping": use_early_stopping,
        "stop_epochs": str(stop_epochs_list) if use_early_stopping else "N/A",
        "avg_accuracy": avg_accuracy, "avg_forgetting": avg_forgetting,
        "avg_bwt": avg_bwt, "avg_fwt": avg_fwt
    }

def print_results_table(results):
    logger.info("\n" + "=" * 180)
    logger.info(" " * 80 + "FINAL EXPERIMENT RESULTS")
    logger.info("=" * 180)
    
    header = (f"| {'Dataset':<20} | {'Buffer':<8} | {'KD Mode':<10} | {'Epochs':<8} | {'Early Stop':<12} | {'Stop Epochs':<25} | {'Avg. Acc (%)':<15} | "
              f"{'Forgetting (%)':<17} | {'BWT (%)':<15} | {'FWT (%)':<15} |")
    separator = (f"|{'-'*22}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*14}|{'-'*27}|{'-'*17}|"
                 f"{'-'*19}|{'-'*17}|{'-'*17}|")
    
    logger.info(header)
    logger.info(separator)

    for result in results:
        row = (f"| {result['dataset']:<20} | {result['buffer_size']:<8} | {result['kd_mode']:<10} | {result['epochs_per_task']:<8} | "
               f"{str(result['early_stopping']):<12} | {result['stop_epochs']:<25} | "
               f"{result['avg_accuracy']:.2f}{'':<11} | "
               f"{result['avg_forgetting']:.2f}{'':<13} | "
               f"{result['avg_bwt']:.2f}{'':<11} | "
               f"{result['avg_fwt']:.2f}{'':<11} |")
        logger.info(row)
        
    logger.info("=" * 180)

# --- Main execution (MODIFIED to save JSON results per dataset) ---
if __name__ == '__main__':
    is_mps_device = torch.backends.mps.is_available()
    datasets_to_run = ["split_cifar10", "split_cifar100"]
    buffer_sizes_to_run = [200, 500, 1000, 2000]
    kd_modes_to_run = ['none', 'all']
    epochs_per_task_options = [5, 10, 100]
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # This list will hold results from all datasets for the final console printout
    final_table_results = []
    
    for dataset in datasets_to_run:
        # This list will hold results for the current dataset to be saved to JSON
        dataset_results = []
        for buffer_size in buffer_sizes_to_run:
            for kd_mode in kd_modes_to_run:
                for epochs in epochs_per_task_options:
                    try:
                        result = run_experiment(
                            dataset_name=dataset, 
                            buffer_size=buffer_size, 
                            kd_mode=kd_mode,
                            epochs_per_task=epochs
                        )
                        dataset_results.append(result)
                        final_table_results.append(result)
                    except Exception as e:
                        logger.error(f"Experiment failed for {dataset}, buffer {buffer_size}, KD {kd_mode}, epochs {epochs}: {e}", exc_info=True)
                        failed_result = {
                            "dataset": dataset, "buffer_size": buffer_size, "kd_mode": kd_mode, 
                            "epochs_per_task": epochs, "early_stopping": (epochs == 100), "stop_epochs": "FAILED",
                            "avg_accuracy": 0.0, "avg_forgetting": 0.0, "avg_bwt": 0.0, "avg_fwt": 0.0
                        }
                        dataset_results.append(failed_result)
                        final_table_results.append(failed_result)

                    finally:
                        if is_mps_device:
                            torch.mps.empty_cache()
                            logger.info(f"Cleared MPS cache after experiment.")
        
        # --- Save results for the completed dataset to a JSON file ---
        json_filename = os.path.join(results_dir, f"{dataset}_results.json")
        logger.info(f"Saving results for dataset '{dataset}' to {json_filename}")
        try:
            with open(json_filename, 'w') as f:
                json.dump(dataset_results, f, indent=4)
            logger.info("Save complete.")
        except Exception as e:
            logger.error(f"Failed to save JSON results for {dataset}: {e}")


    # After all datasets have been processed, print the final consolidated table
    print_results_table(final_table_results)
    
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    sys.exit(0)