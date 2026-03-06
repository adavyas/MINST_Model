# fashion_mnist_simple.py
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# -----------------------------
# Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Seed
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

root = "./data"
full_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

train_size = 50_000
val_size = 10_000
train_set, val_set = random_split(
    full_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

# -----------------------------
# Model
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28 -> 14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14 -> 7

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SmallCNN().to(device)

# -----------------------------
# Loss / Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

# -----------------------------
# Train
# -----------------------------
def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

# -----------------------------
# Training loop
# -----------------------------
epochs = 10
best_val_acc = 0.0
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_fmnist.pt")

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
    )

# -----------------------------
# Test
# -----------------------------
model.load_state_dict(torch.load("checkpoints/best_fmnist.pt", map_location=device))
test_loss, test_acc = evaluate(test_loader)
print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

if device.type == "mps":
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
