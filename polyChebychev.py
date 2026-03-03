import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.utils import to_dense_adj
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Dataset
dataset = WikipediaNetwork(root='data', name='chameleon')
data = dataset[0].to(device)

num_nodes = data.num_nodes
num_features = dataset.num_node_features
num_classes = dataset.num_classes


# Normalized Laplacian
def build_normalized_laplacian(edge_index, num_nodes):
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    A = A.to(device)

    deg = torch.sum(A, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    I = torch.eye(num_nodes, device=device)

    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L

L = build_normalized_laplacian(data.edge_index, num_nodes)

I = torch.eye(num_nodes, device=device)
L_tilde = L - I


# Chebyshev Layer
class ChebLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, K):
        super().__init__()

        self.K = K

        # Chebyshev coefficients
        self.alpha = nn.Parameter(torch.randn(K + 1))

        # 2-layer MLP
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, L_tilde):

        T0 = h
        out = self.alpha[0] * T0

        if self.K >= 1:
            T1 = L_tilde @ h
            out = out + self.alpha[1] * T1

        for k in range(2, self.K + 1):
            Tk = 2 * (L_tilde @ T1) - T0
            out = out + self.alpha[k] * Tk
            T0, T1 = T1, Tk

        # MLP
        x = F.relu(self.lin1(out))
        x = F.relu(self.lin2(x))

        return x


# Full Model
class ChebGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, K):
        super().__init__()
        self.layer1 = ChebLayer(in_dim, hidden_dim, hidden_dim, K)
        self.layer2 = ChebLayer(hidden_dim, hidden_dim, num_classes, K)

    def forward(self, x, L_tilde):
        x = self.layer1(x, L_tilde)
        x = self.layer2(x, L_tilde)
        return x



# Training
num_splits = data.train_mask.shape[1]
all_test_acc = []

for split in range(num_splits):

    print(f"\n========== Split {split} ==========")

    train_mask = data.train_mask[:, split]
    val_mask   = data.val_mask[:, split]
    test_mask  = data.test_mask[:, split]

    model = ChebGNN(
        in_dim=num_features,
        hidden_dim=64,
        num_classes=num_classes,
        K=15
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):

        model.train()
        optimizer.zero_grad()

        out = model(data.x, L_tilde)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            pred = out.argmax(dim=1)

            train_acc = (pred[train_mask] == data.y[train_mask]).float().mean()
            val_acc   = (pred[val_mask] == data.y[val_mask]).float().mean()
            test_acc  = (pred[test_mask] == data.y[test_mask]).float().mean()

            print(f"Epoch {epoch:03d} | "
                  f"Loss {loss:.4f} | "
                  f"Train {train_acc:.4f} | "
                  f"Val {val_acc:.4f} | "
                  f"Test {test_acc:.4f}")

    model.eval()
    out = model(data.x, L_tilde)
    pred = out.argmax(dim=1)
    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

    all_test_acc.append(test_acc)

print("\n====================================")
print(f"Average Test Accuracy: {sum(all_test_acc)/len(all_test_acc):.4f}")