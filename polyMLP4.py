import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.utils import to_dense_adj
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = WikipediaNetwork(root='data', name='chameleon')
data = dataset[0].to(device)

num_nodes = data.num_nodes
num_features = dataset.num_node_features
num_classes = dataset.num_classes

# Build symmetric normalized adjacency
# A_hat = D^{-1/2} A D^{-1/2}
#
def build_normalized_adjacency(edge_index, num_nodes):
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    A = A.to(device)

    deg = torch.sum(A, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat

A_hat = build_normalized_adjacency(data.edge_index, num_nodes)

# Polynomial Adjacency Layer
# m = a0*h + a1*A_hat*h + a2*A_hat^2*h
class AdjPolyLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        # Learnable hop weights
        self.a0 = nn.Parameter(torch.tensor(1.0))
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.a2 = nn.Parameter(torch.tensor(0.0))

        # 2-layer MLP
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, A_hat):
        A1h = A_hat @ h
        A2h = A_hat @ A1h

        m = self.a0 * h + self.a1 * A1h + self.a2 * A2h

        x = F.relu(self.lin1(m))
        x = F.relu(self.lin2(x))
        return x



# Full Model (2 stacked adjacency-poly layers)
class AdjPolyGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = AdjPolyLayer(in_dim, hidden_dim, hidden_dim)
        self.layer2 = AdjPolyLayer(hidden_dim, hidden_dim, num_classes)

    def forward(self, x, A_hat):
        x = self.layer1(x, A_hat)
        x = self.layer2(x, A_hat)
        return x



# Training over all splits
num_splits = data.train_mask.shape[1]
all_test_acc = []

for split in range(num_splits):

    print(f"\n========== Split {split} ==========")

    train_mask = data.train_mask[:, split]
    val_mask   = data.val_mask[:, split]
    test_mask  = data.test_mask[:, split]

    model = AdjPolyGNN(
        in_dim=num_features,
        hidden_dim=64,
        num_classes=num_classes
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, A_hat)

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
    out = model(data.x, A_hat)
    pred = out.argmax(dim=1)

    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
    all_test_acc.append(test_acc)


# final result
print("\n====================================")
print(f"Average Test Accuracy over {num_splits} splits: "
      f"{sum(all_test_acc)/len(all_test_acc):.4f}")