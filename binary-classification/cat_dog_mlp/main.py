
import torch    
import torchvision
from torchvision import datasets
from torchvision import transforms

print(torch.__version__)

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

print("Total images in the dataset:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Dataset classes:", dataset.classes)

# Filter the dataset to include only cat and dog images (labels 3 and 5) but change the label to 0 for cat and 1 for dog
cat_dog_dataset = [(img, 0 if lbl == 3 else 1) for img, lbl in dataset if lbl in [3, 5]]
cat_dog_test_dataset = [(img, 0 if lbl == 3 else 1) for img, lbl in test_dataset if lbl in [3, 5]]

print("Total cat and dog images:", len(cat_dog_dataset))

for i in range(2):
    image, label = cat_dog_dataset[i]
    print(f"Image {i} shape: {image.shape}, Label: {label}")


loader = torch.utils.data.DataLoader(cat_dog_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(cat_dog_test_dataset, batch_size=64, shuffle=False)

image_batch, label_batch = next(iter(loader))
print("Batch image shape:", image_batch.shape)
print("Batch label shape:", label_batch.shape)

class catDogMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(32*32*3, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = catDogMLP()
print(model)

#get the first batch of images and labels
image_batch, label_batch = next(iter(loader))

print("Original batch image shape:", image_batch.shape)
#flatten the images
image_batch = image_batch.view(image_batch.size(0), -1)
print("Flattened batch image shape:", image_batch.shape)

output = model(image_batch)
print("Model output shape:", output.shape)

print("Model output:", output[:5].squeeze())

criterion = torch.nn.BCEWithLogitsLoss()
label_batch = label_batch.float().unsqueeze(1)  # Reshape labels to (batch_size, 1)
loss = criterion(output, label_batch)
print("Loss:", loss.item())


import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
running_loss = 0.0

for epoch in range(epochs):

    for images, labels in loader:

        images = images.view(images.size(0), -1)

        labels = labels.float().unsqueeze(1)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}"
    ) 

# Evaluation on test data
def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the model on test data
    Returns: accuracy and average loss
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Convert output to binary predictions (0 or 1)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = (correct / total) * 100
    
    return accuracy, avg_loss

# Evaluate on test dataset
test_accuracy, test_loss = evaluate_model(model, test_loader, criterion)
print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("="*50 + "\n")
 
total_params = sum(
p.numel()
for p in model.parameters())

print("Total Parameters:", total_params)