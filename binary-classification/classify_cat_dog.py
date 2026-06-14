# ============================================================
# Neural Network Training with Different Variations
# ============================================================
# This program trains a neural network on CIFAR10 dataset with
# different configurations (augmentation, normalization, etc.)
# to improve accuracy. 

# ============================================================
# 1. IMPORT ALL NECESSARY LIBRARIES
# ============================================================
# We import all libraries at the top so it is clear what we need
import torch
from torch import nn, optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import json
import os

# ============================================================
# 2. DEFINE GLOBAL CONSTANTS
# ============================================================
# LOG_FILE stores all experiment results so we can generate a report later
LOG_FILE = "experiment_log.json"


# ============================================================
# 3. HELPER FUNCTIONS FOR REPORT GENERATION
# ============================================================
def generate_report():
    """
    This function creates a summary report of all experiments.
    It reads the logged results, calculates improvement over baseline,
    and saves a CSV file with the results sorted by accuracy.
    """
    # Read the JSON file containing all experiment results
    df = pd.read_json(LOG_FILE)
    
    # Get the accuracy of the first experiment (baseline)
    baseline = df.iloc[0]["accuracy"]
    
    # Calculate how much each experiment improved over the baseline
    df["improvement"] = df["accuracy"] - baseline
    
    # Sort the dataframe so best accuracy appears first
    df = df.sort_values(by="accuracy", ascending=False)
    
    # Save the sorted results to a CSV file (easy to open in Excel)
    df.to_csv("experiment_report.csv", index=False)
    
    # Print the report to the console
    print(df)


# ============================================================
# 4. TRANSFORMATION FUNCTIONS (Data Preprocessing)
# ============================================================
def build_transform(config):
    """
    This function builds the data transformation pipeline based on config.
    Transformations prepare images for the neural network:
    - augmentation: randomly flip/rotate images to help the model learn better
    - normalization: standardize pixel values to a consistent range
    - ToTensor: convert images to PyTorch tensors
    """
    transform_list = []
    
    # If augmentation is enabled, add random transformations
    if config["augmentation"]:
        # Randomly flip images horizontally (left becomes right)
        transform_list.append(transforms.RandomHorizontalFlip())
        # Randomly rotate images by up to 10 degrees
        transform_list.append(transforms.RandomRotation(10))
    
    # Convert image to PyTorch tensor (required for neural networks)
    transform_list.append(transforms.ToTensor())
    
    # If normalization is enabled, standardize pixel values
    if config["normalization"]:
        # Normalize to mean=0.5, std=0.5 for each color channel (R, G, B)
        transform_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    
    # Combine all transformations into one pipeline
    return transforms.Compose(transform_list)


# ============================================================
# 5. DATA LOADER FUNCTIONS
# ============================================================
def create_loaders(transform):
    """
    This function creates train and test data loaders.
    - Dataset: CIFAR10 (10 classes of images: airplane, car, bird, cat, dog, etc.)
    - Loader: batches the data for efficient training
    - shuffle=True: randomly order data during training (helps learning)
    """
    # Create the training dataset (80% of data)
    train_dataset = datasets.CIFAR10(
        root="./data",           # Where to download/store the data
        train=True,              # Use training split
        transform=transform,     # Apply transformations we built
        download=True            # Download if not already present
    )
    
    # Create the test dataset (20% of data)
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,             # Use test split
        transform=transform,
        download=True
    )
    
    # Create data loaders (batches of 64 images at a time)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    return train_loader, test_loader


# ============================================================
# 6. MODEL BUILDER FUNCTIONS
# ============================================================
def build_model(config):
    """
    This function creates and returns the neural network model.
    We use CIFAR10 CNN (not CatDogCNN) because CIFAR10 has 10 classes,
    not just 2 (cat vs dog).
    """
    model = CIFAR10CNN()
    return model


class CIFAR10CNN(torch.nn.Module):
    """
    Convolutional Neural Network for CIFAR10 dataset.
    
    Architecture:
    - 3 Convolutional layers (extract features from images)
    - 2 Fully connected layers (make final classification)
    - Output: 10 classes (airplane, car, bird, cat, dog, horse, ship, truck, etc.)
    """
    def __init__(self):
        super().__init__()
        
        # Convolutional layers (feature extractors)
        self.conv_layers = torch.nn.Sequential(
            # First conv layer: 3 input channels (RGB) -> 16 feature maps
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),                    # Activation function
            torch.nn.MaxPool2d(2),              # Reduce size by half
            
            # Second conv layer: 16 -> 32 feature maps
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Third conv layer: 32 -> 64 feature maps
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Fully connected layers (classifier)
        # After 3 max pools, image size is 64x4x4 (CIFAR10 starts at 32x32)
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),  # Flatten -> 128 units
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)           # 10 output classes for CIFAR10 (FIXED: was 1)
        )
    
    def forward(self, x):
        """
        This defines how data flows through the network.
        Input image -> Conv layers -> Flatten -> FC layers -> Output
        """
        # Pass through convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the 3D tensor to 1D (prepare for fully connected layer)
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x


# ============================================================
# 7. MODEL SAVING FUNCTION
# ============================================================
def save_model(model, config):
    """
    Saves the trained model weights to a file.
    This lets you use the model later without retraining.
    """
    # Create the saved_models directory if it does not exist (FIXED)
    os.makedirs("saved_models", exist_ok=True)
    
    # Save the model weights (not the entire model)
    torch.save(
        model.state_dict(),
        f"saved_models/{config['name']}.pth"
    )


# ============================================================
# 8. TRAINING AND EVALUATION FUNCTION
# ============================================================
def train_and_evaluate(model, train_loader, test_loader, config):
    """
    This is the main training function. It:
    1. Trains the model on training data for specified epochs
    2. Evaluates the model on test data
    3. Returns the test accuracy
    
    Key concepts:
    - Epoch: one complete pass through all training data
    - Loss: how wrong the model is (we want to minimize this)
    - Accuracy: percentage of correct predictions (we want to maximize this)
    """
    # Define what we are measuring (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()
    
    # Define how we update the model (Adam optimizer with learning rate)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop: repeat for each epoch
    for epoch in range(config["epochs"]):
        # Set model to training mode
        model.train()
        
        # Process each batch of training data
        for images, labels in train_loader:
            # Clear previous gradients (needed for new batch)
            optimizer.zero_grad()
            
            # Get predictions from the model
            outputs = model(images)
            
            # Calculate how wrong the predictions are
            loss = criterion(outputs, labels)
            
            # Calculate gradients (how to adjust weights)
            loss.backward()
            
            # Update model weights using the gradients
            optimizer.step()
        
        # Print progress every epoch (helps beginners see training)
        print(f"Epoch {epoch+1}/{config['epochs']} completed")
    
    # Evaluation loop: test on test data
    model.eval()  # Set model to evaluation mode
    
    correct = 0
    total = 0
    
    # No gradient needed for evaluation (faster, less memory)
    with torch.no_grad():
        for images, labels in test_loader:
            # Get predictions
            outputs = model(images)
            
            # Find the class with highest score
            _, predicted = torch.max(outputs.data, 1)
            
            # Count total images and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy as percentage
    accuracy = 100 * correct / total
    
    return accuracy


# ============================================================
# 9. EXPERIMENT LOGGING FUNCTIONS
# ============================================================
def log_experiment(result):
    """
    This function saves experiment results to a JSON file.
    JSON is a simple text format that stores data as dictionaries/lists.
    Each experiment result is appended to the list.
    """
    # Check if the log file already exists
    if os.path.exists(LOG_FILE):
        # Read existing logs
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        # Start with empty list if file does not exist
        logs = []
    
    # Add the new result to the list
    logs.append(result)
    
    # Write the updated list back to the file
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)  # indent=4 makes it readable


# ============================================================
# 10. MAIN EXPERIMENT RUNNER
# ============================================================
def run_experiment(config):
    """
    This function runs one complete experiment with the given configuration.
    Configuration includes: name, epochs, learning_rate, augmentation, normalization
    
    Steps:
    1. Build data transformations
    2. Create train/test data loaders
    3. Build the neural network model
    4. Train and evaluate the model
    5. Log the results
    6. Save the model
    """
    # Step 1: Build transformations based on config
    transform = build_transform(config)
    
    # Step 2: Create data loaders
    train_loader, test_loader = create_loaders(transform)
    
    # Step 3: Build the model
    model = build_model(config)
    
    # Step 4: Train and evaluate (get accuracy)
    accuracy = train_and_evaluate(
        model,
        train_loader,
        test_loader,
        config
    )
    
    # Step 5: Create result dictionary with all important info
    result = {
        "name": config["name"],
        "accuracy": accuracy,
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "augmentation": config["augmentation"],
        "normalization": config["normalization"]
    }
    
    # Step 6: Log the result to JSON file
    log_experiment(result)
    
    # Step 7: Save the trained model
    save_model(model, config)
    
    # Return the result (so we can collect all results)
    return result


# ============================================================
# 11. MAIN EXECUTION: RUN ALL EXPERIMENTS
# ============================================================

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN auto-tuner (faster on GPU)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    
# List to store all experiment results
results = []

# Function to read JSON file (simple lambda function)
file_reading_function = lambda file_path: json.load(open(file_path))

# Load experiment configurations from JSON file
# FIXED: Use forward slash (/) for path, not backslash (\)
EXPERIMENTS = file_reading_function("./experiments.json")

# Run each experiment and collect results
print("Starting experiments...")
for experiment in EXPERIMENTS:
    print(f"Running experiment: {experiment['name']}")
    results.append(run_experiment(experiment))

# Generate the final report after all experiments complete
print("\nGenerating report...")
generate_report()

print("\nAll experiments completed successfully!")