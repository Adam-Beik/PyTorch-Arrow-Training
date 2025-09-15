import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# define arrow classes
ARROW_CLASSES = {
    "forward": [1, 0, 0, 0],
    "left": [0, 1, 0, 0],
    "right": [0, 0, 1, 0],
    "backward": [0, 0, 0, 1],
}


# function to load arrow images from directories along with filename debugging
def load_arrow_images(directory):
    images = []
    labels = []
    filenames = []

    for arrow_type, one_hot in ARROW_CLASSES.items():
        # Find all images for this arrow type
        for filename in os.listdir(directory):
            if filename.startswith(arrow_type) and filename.endswith((".jpg", ".png")):
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Ensure image is 88x88
                if img.shape != (88, 88):
                    img = cv2.resize(img, (88, 88))

                images.append(img)
                labels.append(one_hot)
                filenames.append(filename)

    print(f"Loaded from {directory}: {len(filenames)} images")
    # debug print of first few filenames to verify proper loading
    print(f"Sample filenames: {filenames[:4]}...")
    return np.array(images), np.array(labels)


# load training and testing data
train_images, train_labels = load_arrow_images("arrow_images")
test_images, test_labels = load_arrow_images("unseen_arrow_images")

print(f"Loaded {len(train_images)} training images and {len(test_images)} test images")


# dataset custom class
class ArrowDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# transform to tensor
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# create training dataset
train_dataset = ArrowDataset(train_images, train_labels, transform=transform)

# split training data into train and validation sets, 80/20
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# create testing set
test_dataset = ArrowDataset(test_images, test_labels, transform=transform)

# init dataloaders
batch_size = 4
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# define neural network for arrow recognition
class ArrowNN(nn.Module):
    def __init__(self):
        super(ArrowNN, self).__init__()
        self.fc1 = nn.Linear(88 * 88, 128)  # first hidden layer
        self.fc2 = nn.Linear(128, 64)  # second hidden layer
        self.fc3 = nn.Linear(64, 4)  # output layer

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 88 * 88)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# init model
model = ArrowNN()

# set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # convert  one-hot to class indices for CrossEntropyLoss
            targets = torch.argmax(labels, dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # fwd  pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # bwd pass and optimize
            loss.backward()
            optimizer.step()

            # stats calculation
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                targets = torch.argmax(labels, dim=1)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

    return train_losses, val_losses, train_accuracies, val_accuracies


# training the model and setting number of epochs
num_epochs = 25
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs
)

# evaluating model based  on test set (unseen_arrow_images)
model.eval()
correct = 0
total = 0
class_correct = [0, 0, 0, 0]
class_total = [0, 0, 0, 0]

with torch.no_grad():
    for images, labels in test_loader:
        targets = torch.argmax(labels, dim=1)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # calc per-class accuracy
        c = (predicted == targets).squeeze()
        for i in range(targets.size(0)):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

test_accuracy = 100 * correct / total
print(f"Test Accuracy on Unseen Arrows: {test_accuracy:.2f}%")

# print per-class accuracy
arrow_types = ["forward", "left", "right", "backward"]
for i, arrow_type in enumerate(arrow_types):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Accuracy of {arrow_type}: {accuracy:.2f}%")

# visualize the results via matplotlib
plt.figure(figsize=(12, 5))

# plot loss curves
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

# plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("arrow_training_curves.png")
plt.show()

# save the model curves
torch.save(model.state_dict(), "arrow_classifier_model.pth")
print("Model saved as 'arrow_classifier_model.pth'")

