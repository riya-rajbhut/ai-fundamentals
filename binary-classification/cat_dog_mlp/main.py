import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

print("PyTorch Version:", torch.__version__)

# -----------------------------
# DATASET LOADING
# -----------------------------
transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

print("Total images in CIFAR-10:", len(dataset))
print("Classes:", dataset.classes)

# -----------------------------
# FILTER CATS AND DOGS
# Cat = 0, Dog = 1
# -----------------------------
cat_dog_dataset = []

for img, lbl in dataset:
    if lbl in [3, 5]:
        cat_dog_dataset.append(
            (img, 0 if lbl == 3 else 1)
        )

cat_dog_test_dataset = []

for img, lbl in test_dataset:
    if lbl in [3, 5]:
        cat_dog_test_dataset.append(
            (img, 0 if lbl == 3 else 1)
        )

print("Training Cat/Dog Images:", len(cat_dog_dataset))
print("Testing Cat/Dog Images:", len(cat_dog_test_dataset))

loader = torch.utils.data.DataLoader(
    cat_dog_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    cat_dog_test_dataset,
    batch_size=64,
    shuffle=False
)

image_batch, label_batch = next(iter(loader))

print("Batch Image Shape:", image_batch.shape)
print("Batch Label Shape:", label_batch.shape)


# ==========================================================
# MLP MODEL
# ==========================================================
class CatDogMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 3, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================================
# CNN MODEL
# ==========================================================
class CatDogCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)

        return x


# ==========================================================
# TRAINING FUNCTION
# ==========================================================
def train_model(
    model,
    loader,
    criterion,
    optimizer,
    epochs,
    flatten=False
):

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, labels in loader:

            if flatten:
                images = images.view(
                    images.size(0),
                    -1
                )

            labels = labels.float().unsqueeze(1)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}"
        )


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================
def evaluate_model(
    model,
    test_loader,
    criterion,
    flatten=False
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            if flatten:
                images = images.view(
                    images.size(0),
                    -1
                )

            labels = labels.float().unsqueeze(1)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)

    accuracy = (
        correct / total
    ) * 100

    return accuracy, avg_loss


criterion = torch.nn.BCEWithLogitsLoss()

# ==========================================================
# TRAIN MLP
# ==========================================================
print("\n" + "=" * 50)
print("TRAINING MLP")
print("=" * 50)

mlp_model = CatDogMLP()

optimizer = optim.Adam(
    mlp_model.parameters(),
    lr=0.001
)

train_model(
    mlp_model,
    loader,
    criterion,
    optimizer,
    epochs=30,
    flatten=True
)

mlp_accuracy, mlp_loss = evaluate_model(
    mlp_model,
    test_loader,
    criterion,
    flatten=True
)

print("\nMLP TEST RESULTS")
print(f"Loss     : {mlp_loss:.4f}")
print(f"Accuracy : {mlp_accuracy:.2f}%")

mlp_params = sum(
    p.numel()
    for p in mlp_model.parameters()
)

print("MLP Parameters:", mlp_params)


# ==========================================================
# TRAIN CNN
# ==========================================================
print("\n" + "=" * 50)
print("TRAINING CNN")
print("=" * 50)

cnn_model = CatDogCNN()

optimizer = optim.Adam(
    cnn_model.parameters(),
    lr=0.001
)

train_model(
    cnn_model,
    loader,
    criterion,
    optimizer,
    epochs=30,
    flatten=False
)

cnn_accuracy, cnn_loss = evaluate_model(
    cnn_model,
    test_loader,
    criterion,
    flatten=False
)

print("\nCNN TEST RESULTS")
print(f"Loss     : {cnn_loss:.4f}")
print(f"Accuracy : {cnn_accuracy:.2f}%")

cnn_params = sum(
    p.numel()
    for p in cnn_model.parameters()
)

print("CNN Parameters:", cnn_params)


# ==========================================================
# FINAL COMPARISON
# ==========================================================
print("\n" + "=" * 50)
print("FINAL COMPARISON")
print("=" * 50)

print(
    f"MLP Accuracy : {mlp_accuracy:.2f}%"
)

print(
    f"CNN Accuracy : {cnn_accuracy:.2f}%"
)

print(
    f"Improvement  : "
    f"{cnn_accuracy - mlp_accuracy:.2f}%"
)
