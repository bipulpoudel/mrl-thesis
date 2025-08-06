# Optimize the device using MPS if available - Enhanced for M1 Pro 8 GPU cores
import torch
import os
from logger import logger
import numpy as np
import random
import torch.nn.functional as F

def set_reproducibility_seeds(seed=42):
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set PyTorch CUDA random seed (if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set deterministic behavior for PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic CUDA operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Additional settings for better reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info(f"Reproducibility seeds set to {seed}")

def optimize_device_for_pytorch():
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')        
        # Additional MPS optimizations for M1 Pro
        try:
            # Enable MPS memory management optimizations
            torch.mps.set_per_process_memory_fraction(0.9)
        except Exception as e:
            logger.warning(f'Some MPS optimizations not available: {e}')
            
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'Using CUDA device: {device}')
        logger.info(f'CUDA device name: {torch.cuda.get_device_name(0)}')
        logger.info(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU device: {device}')
        logger.info(f'CPU threads: {torch.get_num_threads()}')
    
    # Set number of threads for CPU operations (useful even with MPS)
    torch.set_num_threads(8)  # Match the number of GPU cores
    
    # Enable optimizations (only if deterministic mode is not set)
    if not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False

    #Disable upper limit on GPU memory
    if device.type == 'mps':
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    logger.info(f"Device optimized for {device}")
    
    return device

# MPS Memory Management
def clear_mps_cache(device, verbose=False):
    """Clear MPS cache and perform garbage collection for better memory management"""
    if device.type == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()  # Ensure all operations complete
        import gc
        gc.collect()  # Python garbage collection
        if verbose:
            logger.info("MPS cache cleared and garbage collection performed")


#Continual learning metrics and utils
class ContinualLearningMetrics:
    def __init__(self):
        self.task_accuracies = []

    def update_task_accuracies(self, task_id, accuracy):
        self.task_accuracies.append((task_id, accuracy))

    def get_final_average_accuracy(self):
        # Get the number of tasks
        num_tasks = len(set(task_id for task_id, _ in self.task_accuracies))
        
        # Create the accuracy matrix
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        # Fill the accuracy matrix
        for task_id, accuracy in self.task_accuracies:
            # Assuming task_id starts from 0
            accuracy_matrix[task_id, task_id] = accuracy
            
            # For tasks after the current task, we can fill with the latest accuracy
            for future_task in range(task_id + 1, num_tasks):
                accuracy_matrix[future_task, task_id] = accuracy
                
        # Calculate average accuracy across all tasks
        avg_accuracy = np.mean([accuracy_matrix[i,i] for i in range(num_tasks)])
        
        return avg_accuracy

    def get_final_average_forgetting(self):
        # Get the number of tasks
        num_tasks = len(set(task_id for task_id, _ in self.task_accuracies))
        
        if num_tasks <= 1:
            return 0.0
            
        # Create accuracy matrix
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        # Fill the accuracy matrix
        for task_id, accuracy in self.task_accuracies:
            accuracy_matrix[task_id, task_id] = accuracy
            for future_task in range(task_id + 1, num_tasks):
                accuracy_matrix[future_task, task_id] = accuracy
                
        # Calculate forgetting for each task
        forgetting = []
        for task in range(num_tasks - 1):  # Exclude last task
            max_acc = max(accuracy_matrix[i, task] for i in range(task, num_tasks))
            final_acc = accuracy_matrix[num_tasks-1, task]
            forgetting.append(max_acc - final_acc)
            
        # Return average forgetting
        return np.mean(forgetting) if forgetting else 0.0
    
    