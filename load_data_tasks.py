from torchvision import transforms
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitMNIST
from utils import optimize_device_for_pytorch, set_reproducibility_seeds
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from torch.utils.data import DataLoader

SEED = 42

def load_data_cifar10_by_tasks(batch_size=32, n_experiences=5, return_datasets=False):
    device = optimize_device_for_pytorch()

    # Set random seeds for reproducibility
    set_reproducibility_seeds(SEED)

    # Define transformations for training and evaluation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Add random crop with padding
        transforms.RandomHorizontalFlip(),      # Add random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # Define dataset root
    datadir = default_dataset_location('cifar10')

    # Create SplitCIFAR10 benchmark
    benchmark = SplitCIFAR10(
        n_experiences=n_experiences,
        dataset_root=datadir,
        return_task_id=True,
        shuffle=True,
        seed=SEED,
        train_transform=transform_train,
        eval_transform=transform_test
    )

    # Pin memory for CUDA
    use_pin_memory = device.type == 'cuda'

    # Create train datasets for each experience
    train_loaders = []
    test_loaders = []
    train_datasets = []  # Store raw datasets for joint training
    test_datasets = []   # Store raw test datasets
    task_classes = []  # Store classes for each task
    total_classes = 0

    for exp in benchmark.train_stream:
        # Get the training dataset for this experience
        train_dataset = exp.dataset

        # Store the classes for this experience
        if exp.classes_in_this_experience is not None:
            task_classes.append(sorted(list(exp.classes_in_this_experience)))
            total_classes += len(exp.classes_in_this_experience)

        # Store raw dataset
        train_datasets.append(train_dataset)

        # Create data loaders for training - no transforms needed since they're already applied
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False,
            pin_memory=use_pin_memory
        )

        train_loaders.append(train_loader)

    # Create test data loaders for each task
    for exp in benchmark.test_stream:
        test_dataset = exp.dataset
        test_datasets.append(test_dataset)
        
        test_loader = DataLoader(
            test_dataset, #type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False,
            pin_memory=use_pin_memory
        )
        test_loaders.append(test_loader)

    if return_datasets:
        return train_datasets, test_datasets, total_classes, task_classes
    else:
        return train_loaders, test_loaders, total_classes, task_classes


def load_data_cifar100_by_tasks(batch_size=32, n_experiences=10, return_datasets=False):
    device = optimize_device_for_pytorch()

    # Set random seeds for reproducibility
    set_reproducibility_seeds(SEED)

    # Define transformations for training and evaluation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Add random crop with padding
        transforms.RandomHorizontalFlip(),      # Add random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Define dataset root
    datadir = default_dataset_location('cifar100')

    # Create SplitCIFAR100 benchmark
    benchmark = SplitCIFAR100(
        n_experiences=n_experiences,
        dataset_root=datadir,
        return_task_id=True,
        shuffle=True,
        seed=SEED,  
        train_transform=transform_train,
        eval_transform=transform_test
    )

    # Pin memory for CUDA
    use_pin_memory = device.type == 'cuda'

    # Create validation datasets for each experience
    train_loaders = []
    test_loaders = []
    train_datasets = []  # Store raw datasets for joint training
    test_datasets = []   # Store raw test datasets
    task_classes = []  # Store classes for each task
    total_classes = 0

    for exp in benchmark.train_stream:
        # Get the training dataset for this experience
        train_dataset = exp.dataset

        # Store the classes for this experience
        if exp.classes_in_this_experience is not None:
            task_classes.append(sorted(list(exp.classes_in_this_experience)))
            total_classes += len(exp.classes_in_this_experience)

        # Store raw dataset
        train_datasets.append(train_dataset)

        # Create data loaders for training - no transforms needed since they're already applied
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False,
            pin_memory=use_pin_memory
        )

        train_loaders.append(train_loader)

    # Create test data loaders for each task
    for exp in benchmark.test_stream:
        test_dataset = exp.dataset
        test_datasets.append(test_dataset)
        
        test_loader = DataLoader(
            test_dataset, #type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False,
            pin_memory=use_pin_memory
        )
        test_loaders.append(test_loader)

    if return_datasets:
        return train_datasets, test_datasets, total_classes, task_classes
    else:
        return train_loaders, test_loaders, total_classes, task_classes


def load_data_split_mnist_by_tasks(batch_size=32, n_experiences=5, return_datasets=False):
    device = optimize_device_for_pytorch()

    # Set random seeds for reproducibility
    set_reproducibility_seeds(SEED)

    # Define transformations for training and evaluation
    transform = transforms.Compose([
        # Repeat the single channel 3 times
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Define dataset root
    datadir = default_dataset_location('mnist')

    # Create SplitMNIST benchmark
    benchmark = SplitMNIST(
        n_experiences=n_experiences,
        dataset_root=datadir,
        return_task_id=True,
        shuffle=True,
        seed=SEED,
        train_transform=transform,
        eval_transform=transform
    )

    # Pin memory for CUDA
    use_pin_memory = device.type == 'cuda'

    # Create validation datasets for each experience
    train_loaders = []
    test_loaders = []
    train_datasets = []  # Store raw datasets for joint training
    test_datasets = []   # Store raw test datasets
    task_classes = []  # Store classes for each task
    total_classes = 0

    for exp in benchmark.train_stream:
        # Get the training dataset for this experience
        train_dataset = exp.dataset

        # Store the classes for this experience
        if exp.classes_in_this_experience is not None:
            task_classes.append(sorted(list(exp.classes_in_this_experience)))
            total_classes += len(exp.classes_in_this_experience)

        # Store raw dataset
        train_datasets.append(train_dataset)

        # Create data loaders for training - no transforms needed since they're already applied
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False,
            pin_memory=use_pin_memory
        )

        train_loaders.append(train_loader)

    # Create test data loaders for each task
    for exp in benchmark.test_stream:
        test_dataset = exp.dataset
        test_datasets.append(test_dataset)
        
        test_loader = DataLoader(
            test_dataset, #type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            persistent_workers=False,
            pin_memory=use_pin_memory
        )
        test_loaders.append(test_loader)

    if return_datasets:
        return train_datasets, test_datasets, total_classes, task_classes
    else:
        return train_loaders, test_loaders, total_classes, task_classes