import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Set Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device {DEVICE}")

# Network Setup
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.05)

    def forward(self, x):
        return self.network_layers(x)

    def train_model(self, train_loader, epochs=50):
        self.to(DEVICE)
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                class_labels = labels.argmax(dim=1)

                self.optimizer.zero_grad()

                # Forward pass
                predictions = self(inputs)

                # Loss
                loss = self.loss_func(predictions, class_labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Print progress
            if epoch % int(0.05 * epochs) == 0:
                print(f"Epoch {epoch + 1} / {epochs} - Loss: {total_loss / len(train_loader):.4f}")

    def test_model(self, test_loader):
        self.to(DEVICE)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                class_labels = labels.argmax(dim=1)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == class_labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')