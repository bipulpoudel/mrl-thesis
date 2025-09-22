import torch
import torch.nn as nn
from tqdm import tqdm
from mrl_resnet_18_model import MRL_ResNet18_CIFAR
from load_data_tasks import load_data_cifar10_by_tasks, load_data_cifar100_by_tasks
from logger import logger
from utils import optimize_device_for_pytorch, set_reproducibility_seeds
import torch.optim as optim
import random
import sys
import numpy as np
import torch.nn.functional as F
import copy

class Hyperparameters:
    EPOCHS_PER_TASK = 5 
    SEED = 42
    DATA_DIR = "./data"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4 
    KD_ALPHA = 0.40
    KD_TEMPERATURE = 2.0
    MRL_DIMENSIONS = [64,128,256,512]
    # Weights for each nested head in the Matryoshka cross-entropy loss.
    # Should sum to 1.
    MRL_WEIGHTS = [1.0,1.0, 1.0, 1.0] 

    # --- NEW: Weights for each nested head in the Knowledge Distillation loss. ---
    # This allows prioritizing distillation from more expressive (larger) heads.
    # The order should correspond to MRL_DIMENSIONS. e.g., [weight_for_256, weight_for_512]
    # Should sum to 1.
    KD_WEIGHTS = [ 1.0, 1.0, 1.0, 1.0]    

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
            if self.n_seen_so_far < self.buffer_size: self.buffer.append(sample)
            else:
                j = random.randint(0, self.n_seen_so_far)
                if j < self.buffer_size: self.buffer[j] = sample
            self.n_seen_so_far += 1
    def get_buffer_size(self): return len(self.buffer)
    def get_samples_from_buffer(self, batch_size, device, current_task_id):
        past_samples = [s for s in self.buffer if s[2] != current_task_id]
        if not past_samples: return None, None
        num_samples_to_get = min(len(past_samples), batch_size)
        random_samples = random.sample(past_samples, num_samples_to_get)
        items, labels, _ = zip(*random_samples)
        items_tensor = torch.stack(items).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        return items_tensor, labels_tensor

# --- Loss Functions (Unchanged) ---
def matroshkya_loss(outputs_list, labels, criterion):
    assert len(outputs_list) == len(Hyperparameters.MRL_WEIGHTS), \
        f"Mismatch between number of MRL outputs ({len(outputs_list)}) and MRL weights ({len(Hyperparameters.MRL_WEIGHTS)})."
    total_loss = sum(
        weight * criterion(outputs, labels)
        for weight, outputs in zip(Hyperparameters.MRL_WEIGHTS, outputs_list)
    )
    return total_loss

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
            outputs_list = model(images)
            outputs = outputs_list[-1] 
            loss = criterion(outputs, labels)   
            val_loss += loss.item() * batch_size 
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += batch_size
    avg_loss = val_loss / val_total if val_total else 0.0
    avg_acc  = 100 * val_correct / val_total if val_total else 0.0
    return avg_loss, avg_acc

# --- UPDATED SECTION: train_model (with Weighted Nested Distillation) ---
def train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, 
                epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=0, teacher_model=None):
    
    tqdm_epochs = tqdm(range(epochs), desc=f"Training Task {task_id+1} (MRL + Weighted Nested KD)")
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
                total_images = torch.cat([current_images, reservoir_images]); total_labels = torch.cat([current_labels, reservoir_labels])
            else:
                total_images = current_images; total_labels = current_labels
            
            optimizer.zero_grad()
            student_outputs_list = model(total_images)
            
            # --- Loss Calculation ---
            # 1. Standard Matryoshka Cross-Entropy Loss
            loss_ce = matroshkya_loss(student_outputs_list, total_labels, criterion)
            
            # 2. Weighted Nested Knowledge Distillation Loss
            loss_kd = 0.0
            if teacher_model is not None and task_id > 0:
                with torch.no_grad():
                    teacher_outputs_list = teacher_model(total_images)

                # Sanity checks
                assert len(student_outputs_list) == len(teacher_outputs_list), \
                    "Student and Teacher must have the same number of MRL heads."
                assert len(student_outputs_list) == len(Hyperparameters.KD_WEIGHTS), \
                    "Mismatch between number of MRL heads and KD weights."

                total_kd_loss = 0.0
                # --- MODIFIED: Apply KD_WEIGHTS to each head's distillation loss ---
                zipped_params = zip(Hyperparameters.KD_WEIGHTS, student_outputs_list, teacher_outputs_list)
                for kd_weight, s_logits, t_logits in zipped_params:
                    total_kd_loss += kd_weight * distillation_loss(
                        s_logits, t_logits, Hyperparameters.KD_TEMPERATURE
                    )
                
                loss_kd = total_kd_loss

            # 3. Combine the losses
            # --- BUG FIX & UPDATE: Correctly combine CE and KD loss ---
            # Standard formulation: (1-alpha) for the main task loss, alpha for the distillation loss.
            loss = (1 - Hyperparameters.KD_ALPHA) * loss_ce + Hyperparameters.KD_ALPHA * loss_kd
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            reservoir_buffer.add_to_buffer(images, labels, task_id_batch)

            # --- Metrics Update ---
            full_dim_student_logits = student_outputs_list[-1]
            train_loss += loss.item()
            train_correct += (full_dim_student_logits.argmax(dim=1) == total_labels).sum().item()
            train_total += total_labels.size(0)
            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            tqdm_train_loader.set_description(f"Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}")
        
        scheduler.step()
        tqdm_epochs.set_description(f"Task {task_id+1} - Epoch {epoch+1}/{epochs} - Loss: {current_loss:.4f}, Acc: {batch_acc:.2f}%")

# --- run_experiment (Updated logging) ---
def run_experiment(dataset_name, buffer_size):
    set_reproducibility_seeds(Hyperparameters.SEED)
    device = optimize_device_for_pytorch()

    # --- UPDATED: Mode string for clarity ---
    mode_str = "MRL + Weighted Nested KD"
    logger.info(f"{'='*70}\nSTARTING EXPERIMENT: Dataset={dataset_name}, Buffer={buffer_size}, Mode=({mode_str})\n{'='*70}")

    if dataset_name == "split_cifar10": train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    elif dataset_name == "split_cifar100": train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks(batch_size=Hyperparameters.BATCH_SIZE)
    else: raise ValueError(f"Invalid dataset: {dataset_name}")

    num_tasks = len(train_loaders)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))

    logger.info(f"Initializing MRL_ResNet18_CIFAR with dims: {Hyperparameters.MRL_DIMENSIONS}")
    model = MRL_ResNet18_CIFAR(
        num_classes=num_classes, 
        pretrained=True, 
        mrl_dims=Hyperparameters.MRL_DIMENSIONS
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    baseline_accuracies = []
    for task_id in range(num_tasks):
        _, test_acc = evaluate_model(model, device, test_loaders[task_id], criterion)
        baseline_accuracies.append(test_acc)
    
    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY, nesterov=True)
    reservoir_buffer = ReservoirBuffer(buffer_size=buffer_size)
    teacher_model = None
    
    for i, train_loader in enumerate(train_loaders):
        logger.info(f"{'-'*60}\nTraining on Task {i+1}/{num_tasks} (Classes: {task_classes[i]})\n{'-'*60}")
        if i > 0: teacher_model = copy.deepcopy(model); teacher_model.eval()
        for param_group in optimizer.param_groups: param_group['lr'] = Hyperparameters.LEARNING_RATE
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Hyperparameters.EPOCHS_PER_TASK, eta_min=1e-4)
        train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler,
                    epochs=Hyperparameters.EPOCHS_PER_TASK, task_id=i, teacher_model=teacher_model)
        for j in range(num_tasks):
            _, test_acc = evaluate_model(model, device, test_loaders[j], criterion)
            accuracy_matrix[i, j] = test_acc
        logger.info(f"Accuracies after task {i+1}: {[f'{acc:.2f}%' for acc in accuracy_matrix[i]]}")

    logger.info(f"{'='*60}\nFINAL METRICS for Mode: ({mode_str})\n{'='*60}")
    final_accuracies = accuracy_matrix[-1, :]
    avg_accuracy = np.mean(final_accuracies)
    bwt_list = [accuracy_matrix[num_tasks-1, j] - accuracy_matrix[j, j] for j in range(num_tasks - 1)]
    avg_bwt = np.mean(bwt_list) if bwt_list else 0.0
    forgetting_list = [np.max(accuracy_matrix[j:, j]) - accuracy_matrix[num_tasks - 1, j] for j in range(num_tasks - 1)]
    avg_forgetting = np.mean(forgetting_list) if forgetting_list else 0.0
    fwt_list = [accuracy_matrix[i-1, i] - baseline_accuracies[i] for i in range(1, num_tasks)]
    avg_fwt = np.mean(fwt_list) if fwt_list else 0.0
    logger.info(f"Final Accuracy Matrix:\n{np.round(accuracy_matrix, 2)}")
    logger.info(f"--------------------------------------------------\nFINAL AVERAGE ACCURACY: {avg_accuracy:.2f}%")
    
    return {"dataset": dataset_name, "buffer_size": buffer_size, "mode": mode_str,
            "avg_accuracy": avg_accuracy, "avg_forgetting": avg_forgetting, "avg_bwt": avg_bwt, "avg_fwt": avg_fwt}

# --- print_results_table (Updated to reflect new mode name) ---
def print_results_table(results):
    logger.info("\n" + "=" * 140)
    logger.info(" " * 60 + "FINAL EXPERIMENT RESULTS")
    logger.info("=" * 140)
    header = (f"| {'Dataset':<25} | {'Buffer':<8} | {'Mode':<28} | {'Avg. Accuracy (%)':<20} | "
              f"{'Forgetting (%)':<17} | {'BWT (%)':<15} | {'FWT (%)':<15} |")
    logger.info(header); logger.info(f"|{'-'*27}|{'-'*10}|{'-'*30}|{'-'*22}|{'-'*19}|{'-'*17}|{'-'*17}|")
    for result in results:
        row = (f"| {result['dataset']:<25} | {result['buffer_size']:<8} | {result['mode']:<28} | "
               f"{result['avg_accuracy']:.2f}{'':<16} | {result['avg_forgetting']:.2f}{'':<13} | "
               f"{result['avg_bwt']:.2f}{'':<11} | {result['avg_fwt']:.2f}{'':<11} |")
        logger.info(row)
    logger.info("=" * 140)

# --- Main execution block (Unchanged) ---
if __name__ == '__main__':
    is_mps_device = torch.backends.mps.is_available()
    datasets_to_run = ["split_cifar10"]
    buffer_sizes_to_run = [200, 500, 1000, 2000]
    
    all_results = []
    
    for dataset in datasets_to_run:
        for buffer_size in buffer_sizes_to_run:
            try:
                result = run_experiment(
                    dataset_name=dataset, 
                    buffer_size=buffer_size
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed for {dataset}, buffer {buffer_size}: {e}", exc_info=True)
                all_results.append({
                    "dataset": dataset, "buffer_size": buffer_size, "mode": "MRL + W-Nested KD (Failed)",
                    "avg_accuracy": 0.0, "avg_forgetting": 0.0, "avg_bwt": 0.0, "avg_fwt": 0.0
                })
            finally:
                if is_mps_device: torch.mps.empty_cache(); logger.info(f"Cleared MPS cache.")
                    
    print_results_table(all_results)
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    sys.exit(0)