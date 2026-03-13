import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures

# ======================================
# Load Chameleon dataset
# ======================================

dataset = WikipediaNetwork(
    root="data/WikipediaNetwork",
    name="chameleon",
    transform=NormalizeFeatures()
)

data = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

row, col = data.edge_index

# ======================================
# Structural Features
# ======================================

article_length = (data.x > 0).sum(dim=1, keepdim=True).float()
article_length = article_length / article_length.max()

feature_magnitude = data.x.abs().sum(dim=1, keepdim=True)
feature_magnitude = feature_magnitude / feature_magnitude.max()

deg = torch.bincount(row, minlength=data.num_nodes).float().unsqueeze(1)
deg = deg / deg.max()

in_deg = torch.bincount(col, minlength=data.num_nodes).float().unsqueeze(1)
in_deg = in_deg / in_deg.max()

log_deg = torch.log1p(torch.bincount(row, minlength=data.num_nodes).float()).unsqueeze(1)
log_deg = log_deg / log_deg.max()

# ======================================
# PageRank
# ======================================

N = data.num_nodes
pr = torch.ones(N, device=device) / N
damping = 0.85

out_deg = torch.bincount(row, minlength=N).float().to(device)
out_deg[out_deg == 0] = 1

for _ in range(50):

    pr_new = torch.zeros_like(pr)

    pr_new.index_add_(0, col, pr[row] / out_deg[row])

    pr = (1 - damping) / N + damping * pr_new

pagerank = pr.unsqueeze(1)
pagerank = pagerank / pagerank.max()

# ======================================
# Spectral Eigenvector Features
# ======================================

print("Computing Laplacian eigenvectors...")

N = data.num_nodes

A = torch.zeros((N, N), device=device)
A[row, col] = 1
A[col, row] = 1

deg_vec = A.sum(dim=1)

D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg_vec + 1e-10))

I = torch.eye(N, device=device)

L = I - D_inv_sqrt @ A @ D_inv_sqrt

eigvals, eigvecs = torch.linalg.eigh(L)

spectral_features = eigvecs[:, 1:17]

spectral_features = spectral_features / spectral_features.abs().max()

# ======================================
# Concatenate features
# ======================================

extra_features = torch.cat([
    article_length,
    feature_magnitude,
    deg,
    in_deg,
    log_deg,
    pagerank,
    spectral_features
], dim=1)

data.x = torch.cat([data.x, extra_features], dim=1)

print("New feature dimension:", data.x.shape[1])

split = 1
train_mask = data.train_mask[:, split]
val_mask   = data.val_mask[:, split]
test_mask  = data.test_mask[:, split]

# ======================================
# Edge Navier–Stokes Layer
# ======================================

class EdgeNavierStokesLayer(nn.Module):

    def __init__(self, hidden_dim, dt=0.03):
        super().__init__()

        self.dt = dt

        # viscosity
        self.viscosity_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # PRESSURE FROM DIFFERENCE
        self.pressure_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # learned force
        self.force_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, edge_index):

        row, col = edge_index

        hi = h[row]
        hj = h[col]

        edge_input = torch.cat([hi, hj], dim=1)

        # diffusion
        nu = self.viscosity_mlp(edge_input)
        diffusion = nu * (hj - hi)

        # learned force
        force = self.force_mlp(edge_input)

        # pressure gradient
        pressure = self.pressure_mlp(hi - hj)

        message = diffusion + force - pressure

        agg = torch.zeros_like(h)

        agg.index_add_(0, row, message)

        return h + self.dt * agg


# ======================================
# Full GNN
# ======================================

class EdgeNavierStokesGNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, layers=4):

        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EdgeNavierStokesLayer(hidden_dim)
            for _ in range(layers)
        ])

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):

        h = self.input_proj(x)

        h0 = h

        for layer in self.layers:
            h = layer(h, edge_index)

        h = h + h0

        return self.classifier(h)


# ======================================
# Initialize model
# ======================================

model = EdgeNavierStokesGNN(
    in_dim=data.x.shape[1],
    hidden_dim=96,
    out_dim=dataset.num_classes,
    layers=4
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20
)

# ======================================
# Training
# ======================================

def train():

    model.train()

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.cross_entropy(out[train_mask], data.y[train_mask])

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(mask):

    model.eval()

    out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)

    acc = (pred[mask] == data.y[mask]).float().mean()

    loss = F.cross_entropy(out[mask], data.y[mask])

    return loss.item(), acc.item()


# ======================================
# Train Loop
# ======================================

best_test_acc = 0

for epoch in range(1, 161):

    train_loss = train()

    val_loss, val_acc = evaluate(val_mask)

    scheduler.step(val_loss)

    test_loss, test_acc = evaluate(test_mask)

    best_test_acc = max(best_test_acc, test_acc)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Test Acc {test_acc:.4f}"
    )

print("\nBest Test Accuracy:", best_test_acc)