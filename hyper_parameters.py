# Hyperparameters for the Thesis

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