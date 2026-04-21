import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Custom Prunable Linear Layer ──────────────────────────────────────────────
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Standard weight and bias (just like nn.Linear)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight, also learnable
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        # Sigmoid squashes gate_scores to (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiply: "prune" weak weights
        pruned_weights = self.weight * gates

        # Standard linear operation: x @ W.T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        # Helper to read gate values (used for sparsity calculation)
        return torch.sigmoid(self.gate_scores)


# ── Neural Network using PrunableLinear ───────────────────────────────────────
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR-10 images are 32x32x3 = 3072 input features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)   # 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)   # Flatten image → 3072
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)              # Raw logits (no softmax needed)
        return x

    def get_all_gates(self):
        # Returns all gate values from every PrunableLinear layer
        gates = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates.append(layer.get_gates().detach().flatten())
        return torch.cat(gates)