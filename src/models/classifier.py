import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class SimpleMNISTConvClassifier(nn.Module):
    def __init__(self, weights_path=None, num_classes=10):
        self.num_classes = num_classes
        super(SimpleMNISTConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        if weights_path:
            print("Loading pretrained weights from:", weights_path)
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 32, 28, 28)
        x = self.pool(x)  # (batch, 32, 14, 14)
        x = F.relu(self.conv2(x))  # (batch, 64, 14, 14)
        x = self.pool(x)  # (batch, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example usage:
# model = SimpleMNISTConvClassifier()

# Add simple training with saved of the model weights and with evaluation on MNIST dataset

if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss function, optimizer
    model = SimpleMNISTConvClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}"
        )
        # Evaluation on the validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f"Accuracy on validation set after epoch {epoch+1}: {100 * correct / total:.2f}%"
        )
        model.train()

    # Save the trained model weights
    import os

    os.makedirs("../../src/pretrained_models", exist_ok=True)
    torch.save(model.state_dict(), "../../src/pretrained_models/mnist_classifier.pth")
    print("Model weights saved to src/pretrained_models/mnist_classifier.pth")
