# Optimize the device using MPS if available - Enhanced for M1 Pro 8 GPU cores
import torch
import os
from logger import logger
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn

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


class MatryoshkaLoss(nn.Module):
    """
    Calculates the total loss for a Matryoshka-style model.
    It takes a list of predictions (one for each head) and computes the
    sum of their individual losses against the same target.
    """
    def __init__(self):
        super(MatryoshkaLoss, self).__init__()
        self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, mrl_outputs, targets):
        """
        Args:
            mrl_outputs (list of Tensors): The list of output logits from each MRL head.
            targets (Tensor): The ground truth labels.
        
        Returns:
            Tensor: The total, aggregated loss.
        """
        total_loss = 0.0
        for output in mrl_outputs:
            total_loss += self.base_criterion(output, targets)
        
        return total_loss