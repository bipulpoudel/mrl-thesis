import torch
import torch.nn as nn
from tqdm import tqdm
from resnet_18_model import ResNet18_CIFAR
from load_data_tasks import load_data_cifar10_by_tasks, load_data_cifar100_by_tasks, load_data_split_mnist_by_tasks
from logger import logger
from utils import optimize_device_for_pytorch
import torch.optim as optim
import random
import sys

class Hyperparameters:
    EPOCHS_PER_TASK = 5

    #Seed
    SEED = 42

    #Data directory
    DATA_DIR = "./data"

    #Batch sizes and number of workers
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    #Number of classes, MODIFY THIS SPECIFICALLY FOR THE CIFAR-10 DATASET
    NUM_CLASSES = 10

    #Learning rate
    LEARNING_RATE = 0.001

    #Momentum
    MOMENTUM = 0.9

    #Weight decay
    WEIGHT_DECAY = 5e-4 

    #Buffer size
    BUFFER_SIZE = 1000

#Reservoir Buffer for replay in the replay buffer
class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.n_seen_so_far = 0 # Counter for total samples seen

    def add_to_buffer(self, items, labels):
        """
        Adds samples to the buffer using reservoir sampling.
        'items' can be raw images or feature embeddings.
        """
        for item, label in zip(items, labels):
            # Detach and move to CPU as before
            sample = (item.detach().cpu(), label.detach().cpu())
            
            if self.n_seen_so_far < self.buffer_size:
                # If the buffer is not full, just add the new sample
                self.buffer.append(sample)
            else:
                # If the buffer is full, decide whether to replace an old sample
                # The probability of replacement is buffer_size / n_seen_so_far
                j = random.randint(0, self.n_seen_so_far)
                if j < self.buffer_size:
                    # Replace a random existing sample
                    self.buffer[j] = sample
            
            self.n_seen_so_far += 1

    def get_buffer_size(self):
        return len(self.buffer)
    
    def get_samples_from_buffer(self, batch_size, device):
        """
        This method remains the same. It just gets a random sample from
        whatever is currently in the buffer.
        """
        num_samples_to_get = min(len(self.buffer), batch_size)
        
        if num_samples_to_get == 0:
            return None, None

        random_samples = random.sample(self.buffer, num_samples_to_get)
        items, labels = zip(*random_samples)

        items_tensor = torch.stack(items).to(device)
        labels_tensor = torch.stack(labels).to(device)

        return items_tensor, labels_tensor

def evaluate_model(model, device, validation_loader, criterion):
    model.eval()

    #Initialize the validation loss
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, (images, labels, task_id) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)   

            #Update the validation loss
            val_loss += loss.item()

            #Predict the class
            predicted = torch.argmax(outputs, dim=1)
            #Update the validation correct
            val_correct += (predicted == labels).sum().item()

            #Update the validation total
            val_total += labels.size(0)

    #Calculate the validation loss
    val_loss = val_loss / len(validation_loader)
    val_acc = 100 * val_correct / val_total

    return val_loss, val_acc


def train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, epochs=Hyperparameters.EPOCHS_PER_TASK):
    #Train the model
    tqdm_epochs = tqdm(range(epochs), desc="Training")
    for epoch in tqdm_epochs:
        model.train()

        #Initialize the training loss
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        #Train the model
        tqdm_train_loader = tqdm(train_loader, desc="Progress:", leave=False)
        for i, (images, labels, task_id) in enumerate(tqdm_train_loader):
            current_images, current_labels = images.to(device), labels.to(device)

            #Get buffer size    
            buffer_size = reservoir_buffer.get_buffer_size()

            #Get the samples from the reservoir buffer
            if buffer_size > 0:
                reservoir_images, reservoir_labels = reservoir_buffer.get_samples_from_buffer(batch_size=Hyperparameters.BATCH_SIZE, device=device)
                total_images = torch.cat([current_images, reservoir_images])
                total_labels = torch.cat([current_labels, reservoir_labels])
            else:
                total_images = current_images
                total_labels = current_labels

            #Zero the gradients
            optimizer.zero_grad()

            #Forward pass
            outputs = model(total_images)

            #Calculate the loss
            loss = criterion(outputs, total_labels)

            #Backward pass
            loss.backward()

            #Update the weights
            optimizer.step()

            #Update the training loss
            train_loss += loss.item()

            #Update the training correct
            train_correct += (outputs.argmax(dim=1) == total_labels).sum().item()

            #Update the training total
            train_total += total_labels.size(0)

            # Calculate current batch accuracy and update progress bar
            batch_acc = 100 * train_correct / train_total
            current_loss = train_loss / (i + 1)
            
            # Update batch progress bar with current metrics
            tqdm_train_loader.set_description(
                f"Batch Loss: {current_loss:.4f}, Batch Acc: {batch_acc:.2f}%, LR: {current_lr:.4f}"
            )
        #Update the learning rate
        scheduler.step()

# Example usage
if __name__ == '__main__':

    #selected dataset
    dataset = "split_cifar10"
    num_classes = 10 #default number of classes

    #Optimize the device for pytorch
    device = optimize_device_for_pytorch()


    logger.info("=" * 60)
    logger.info(f"STARTING BASIC PIPELINE FOR {dataset}")
    logger.info("=" * 60)

    if dataset == "split_cifar10":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar10_by_tasks()
        num_classes = 10
    elif dataset == "split_cifar100":
        train_loaders, test_loaders, num_classes, task_classes = load_data_cifar100_by_tasks()
        num_classes = 100
    elif dataset == "split_mnist":
        train_loaders, test_loaders, num_classes, task_classes = load_data_split_mnist_by_tasks()
        num_classes = 10
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    #Load the resnet18 model
    model = ResNet18_CIFAR(num_classes=num_classes, pretrained=True)
    #Modify the final fully connected layer for the number of classes, this is important for the model to be able to run on the GPU
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    #Move the model to the device, this is important for the model to be able to run on the GPU
    model.to(device)

    # Define optimizer and scheduler HERE, only once
    optimizer = optim.SGD(model.parameters(), lr=Hyperparameters.LEARNING_RATE, 
                        momentum=Hyperparameters.MOMENTUM, weight_decay=Hyperparameters.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    #Initialize the reservoir buffer
    reservoir_buffer = ReservoirBuffer(buffer_size=Hyperparameters.BUFFER_SIZE)

    for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        logger.info("=" * 60)
        logger.info(f"Training on task {i+1} of {len(train_loaders)}")
        logger.info("=" * 60)
        #Define the loss function
        criterion = nn.CrossEntropyLoss()

        #Train the model
        train_model(model, device, train_loader, criterion, reservoir_buffer, optimizer, scheduler, epochs=Hyperparameters.EPOCHS_PER_TASK)

        #Add the samples to the reservoir buffer
        logger.info("Adding samples to the reservoir buffer")
        for batch_idx, (images, labels, task_id) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            reservoir_buffer.add_to_buffer(images, labels)

        #Evaluate the model on current task and all previous tasks
        average_test_acc = 0.0
        #Loop through all previous tasks
        for j in range(i + 1):
            logger.info("=" * 60)
            logger.info(f"Model Evaluation for task {j+1}")
            logger.info("=" * 60)

            #Get the test loader for the current task
            test_loader_current_task = test_loaders[j]

            #After training has completed, test the model
            test_loss, test_acc = evaluate_model(model, device, test_loader_current_task, criterion)
            logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

            average_test_acc += test_acc

        average_test_acc = average_test_acc / (i + 1)
        logger.info(f"Average accuracy for all tasks so far: {average_test_acc:.2f}%")


    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!!!")
    logger.info("=" * 60)

    #Close the process
    sys.exit(0)
    
    
    