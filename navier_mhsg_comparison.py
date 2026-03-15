import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, WebKB, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import get_laplacian, add_self_loops, remove_self_loops, coalesce
from torch_geometric.data import Data
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# ==========================================
# 1. Deduplication & Data Engineering
# ==========================================
def deduplicate_graph(data):
    """Removes identical nodes to prevent train/test data leakage."""
    unique_vals, inverse_indices = torch.unique(data.x, dim=0, return_inverse=True)
    num_unique = unique_vals.shape[0]
    
    if num_unique == data.num_nodes: return data
        
    first_occurrence = torch.zeros(num_unique, dtype=torch.long, device=data.x.device)
    arange = torch.arange(data.num_nodes, dtype=torch.long, device=data.x.device)
    first_occurrence.scatter_(0, inverse_indices.flip(0), arange.flip(0))
    
    new_x, new_y = data.x[first_occurrence], data.y[first_occurrence]
    
    if data.train_mask.dim() > 1:
        new_train_mask, new_val_mask, new_test_mask = data.train_mask[first_occurrence, 0], data.val_mask[first_occurrence, 0], data.test_mask[first_occurrence, 0]
    else:
        new_train_mask, new_val_mask, new_test_mask = data.train_mask[first_occurrence], data.val_mask[first_occurrence], data.test_mask[first_occurrence]

    new_edge_index = inverse_indices[data.edge_index]
    new_edge_index, _ = remove_self_loops(new_edge_index)
    new_edge_index = coalesce(new_edge_index)
    
    return Data(x=new_x, edge_index=new_edge_index, y=new_y, train_mask=new_train_mask, val_mask=new_val_mask, test_mask=new_test_mask)

def compute_mhsg_matrices(edge_index, num_nodes, device):
    edge_index_self, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_self
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_adj_weight = deg_inv_sqrt[row] * 1.0 * deg_inv_sqrt[col]
    A_norm = torch.sparse_coo_tensor(edge_index_self, norm_adj_weight, (num_nodes, num_nodes)).to(device)
    
    edge_index_lap, edge_weight_lap = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    L_norm = torch.sparse_coo_tensor(edge_index_lap, edge_weight_lap, (num_nodes, num_nodes)).to(device)
    return A_norm, L_norm

# ==========================================
# 2. Architectures
# ==========================================
class EdgeNavierStokesLayer(nn.Module):
    def __init__(self, hidden_dim, dt=0.03):
        super().__init__()
        self.dt = dt
        self.viscosity_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.pressure_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.force_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h, edge_index):
        row, col = edge_index
        edge_input = torch.cat([h[row], h[col]], dim=1)
        nu = self.viscosity_mlp(edge_input)
        diffusion = nu * (h[col] - h[row])
        message = diffusion + self.force_mlp(edge_input) - self.pressure_mlp(edge_input)
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, message)
        return h + self.dt * agg

class EdgeNavierStokesGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EdgeNavierStokesLayer(hidden_dim) for _ in range(layers)])
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, *args): 
        h = self.input_proj(x)
        h0 = h
        for layer in self.layers: h = layer(h, edge_index)
        return self.classifier(h + h0)

class MultiHopSpectralGating(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, K_hops=2, dropout=0.5):
        super().__init__()
        self.K_hops = K_hops
        self.feature_proj = nn.Linear(in_dim, hidden_dim)
        total_channels = 1 + (2 * K_hops)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * total_channels, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, A_norm, L_norm):
        h_0 = F.relu(self.feature_proj(x))
        signals = [h_0]
        h_curr_low, h_curr_high = h_0, h_0
        for _ in range(self.K_hops):
            h_curr_low = torch.sparse.mm(A_norm, h_curr_low)
            h_curr_high = torch.sparse.mm(L_norm, h_curr_high)
            signals.extend([h_curr_low, h_curr_high])
        z = torch.cat(signals, dim=1)
        return self.gate_mlp(z)

# ==========================================
# 3. Visualization
# ==========================================
def plot_benchmark_results(results, datasets):
    ns_scores = [results[ds]['Navier-Stokes'] for ds in datasets]
    mhsg_scores = [results[ds]['MHSG'] for ds in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, ns_scores, width, label='Navier-Stokes (Low-Pass)', color='#3498db')
    rects2 = ax.bar(x + width/2, mhsg_scores, width, label='MHSG (High/Low-Pass)', color='#e74c3c')

    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Architecture Performance Across Heterophilic Datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('architecture_benchmark.png', dpi=150)
    print("\nVisualization saved as 'architecture_benchmark.png'")

# ==========================================
# 4. Training Loop & Runner
# ==========================================
def train_and_eval(model, data, A_norm, L_norm, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    split = 0
    train_mask, val_mask, test_mask = data.train_mask[:, split] if data.train_mask.dim() > 1 else data.train_mask, data.val_mask[:, split] if data.val_mask.dim() > 1 else data.val_mask, data.test_mask[:, split] if data.test_mask.dim() > 1 else data.test_mask
    best_test_acc = 0.0

    pbar = tqdm(range(epochs), leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, A_norm, L_norm)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, A_norm, L_norm)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
            
            if val_acc > best_test_acc: best_test_acc = test_acc
            pbar.set_postfix({'Val': f"{val_acc*100:.1f}%", 'Test': f"{test_acc*100:.1f}%"})

    return best_test_acc * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Benchmark on {device}...\n")
    
    datasets_to_test = ['Chameleon', 'Texas', 'Roman-empire', 'Amazon-ratings', 'Minesweeper']
    results = {ds: {} for ds in datasets_to_test}

    dataset_loaders = {
        'Chameleon': WikipediaNetwork(root="data/WikipediaNetwork", name="chameleon", transform=NormalizeFeatures()),
        'Texas': WebKB(root="data/WebKB", name="texas", transform=NormalizeFeatures()),
        'Roman-empire': HeterophilousGraphDataset(root="data/Heterophilous", name="Roman-empire", transform=NormalizeFeatures()),
        'Amazon-ratings': HeterophilousGraphDataset(root="data/Heterophilous", name="Amazon-ratings", transform=NormalizeFeatures()),
        'Minesweeper': HeterophilousGraphDataset(root="data/Heterophilous", name="Minesweeper", transform=NormalizeFeatures())
    }

    for ds_name in datasets_to_test:
        print(f"[{datasets_to_test.index(ds_name)+1}/{len(datasets_to_test)}] Processing {ds_name}...")
        dataset = dataset_loaders[ds_name]
        data = dataset[0].to(device)
        
        # Deduplicate Chameleon/Texas
        if ds_name in ['Chameleon', 'Texas']:
            data = deduplicate_graph(data)

        # Precompute Spectral Matrices
        A_norm, L_norm = compute_mhsg_matrices(data.edge_index, data.num_nodes, device)

        in_dim, out_dim = data.x.shape[1], dataset.num_classes

        # 1. Navier-Stokes
        print("    -> Training Navier-Stokes GNN")
        ns_model = EdgeNavierStokesGNN(in_dim, 128, out_dim, layers=4).to(device)
        results[ds_name]['Navier-Stokes'] = train_and_eval(ns_model, data, A_norm, L_norm)

        # 2. Multi-Hop Spectral Gating
        print("    -> Training Multi-Hop Spectral Gating")
        mhsg_model = MultiHopSpectralGating(in_dim, 64, out_dim, K_hops=2).to(device)
        results[ds_name]['MHSG'] = train_and_eval(mhsg_model, data, A_norm, L_norm)

        # Memory Cleanup
        del ns_model, mhsg_model, data, dataset, A_norm, L_norm
        torch.cuda.empty_cache()
        gc.collect()

    plot_benchmark_results(results, datasets_to_test)

if __name__ == "__main__":
    main()