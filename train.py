import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import SelfPruningNet

# ── Load CIFAR-10 Data ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=128, shuffle=False)


# ── Training Function ─────────────────────────────────────────────────────────
def train(lam, epochs=10):
    model     = SelfPruningNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== Training with λ = {lam} ===")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)

            # Classification loss
            cls_loss = criterion(outputs, labels)

            # Sparsity loss — L1 norm of all gate values
            sparsity_loss = model.get_all_gates().sum()

            # Total loss
            loss = cls_loss + lam * sparsity_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f}")

    return model


# ── Evaluation Function ───────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Sparsity — % of gates below threshold 0.01
    gates    = model.get_all_gates().numpy()
    sparsity = 100 * (gates < 0.01).sum() / len(gates)

    return accuracy, sparsity, gates


# ── Run Experiments for 3 Lambda Values ──────────────────────────────────────
lambdas  = [0.0001, 0.001, 0.01]
results  = []
best_model, best_gates = None, None
best_acc = 0

for lam in lambdas:
    model = train(lam, epochs=10)
    accuracy, sparsity, gates = evaluate(model)
    results.append((lam, accuracy, sparsity))
    print(f"Lambda={lam}  Accuracy={accuracy:.2f}%  Sparsity={sparsity:.2f}%")

    if accuracy > best_acc:
        best_acc   = accuracy
        best_model = model
        best_gates = gates


# ── Print Final Results Table ─────────────────────────────────────────────────
print("\n\nFINAL RESULTS:")
print(f"{'Lambda':<10} {'Test Accuracy (%)':<22} {'Sparsity Level (%)'}")
print("-" * 52)
for lam, acc, spar in results:
    print(f"{lam:<10} {acc:<22.2f} {spar:.2f}")


# ── Plot Gate Distribution for Best Model ────────────────────────────────────
best_lam = results[[r[1] for r in results].index(max(r[1] for r in results))][0]
plt.figure(figsize=(8, 5))
plt.hist(best_gates, bins=50, color='steelblue', edgecolor='black')
plt.title(f'Gate Value Distribution — Best Model (λ={best_lam})')
plt.xlabel('Gate Value')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('gate_distribution.png')
plt.show()
print("\nPlot saved as gate_distribution.png ✅")