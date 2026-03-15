import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, WebKB, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.data import Data

# ==========================================
# 1. Graph Deduplication 
# ==========================================
def deduplicate_graph(data):
    """Removes identical nodes to prevent train/test data leakage."""
    unique_vals, inverse_indices = torch.unique(data.x, dim=0, return_inverse=True)
    num_unique = unique_vals.shape[0]
    
    if num_unique == data.num_nodes:
        return data # No duplicates
        
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

# ==========================================
# 2. Baseline Models

class GCN_Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

class GAT_Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)

class H2GCN_Net(nn.Module):
    """Simplified H2GCN capturing 1-hop and 2-hop distances independently."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim, normalize=True, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=True, add_self_loops=False)
        self.lin_final = nn.Linear(hidden_dim * 3, out_dim) # Concatenates h, Ah, A^2h

    def forward(self, x, edge_index):
        h0 = F.relu(self.lin1(x))
        h0 = F.dropout(h0, p=0.5, training=self.training)
        
        h1 = self.conv1(h0, edge_index) # 1-hop
        h2 = self.conv2(h1, edge_index) # 2-hop
        
        h_concat = torch.cat([h0, h1, h2], dim=1)
        return self.lin_final(h_concat)

# ==========================================
# 3. Benchmark Execution
# ==========================================
def train_and_eval(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_test_acc = 0.0
    
    # Handle mask shapes
    train_mask = data.train_mask[:, 0] if data.train_mask.dim() > 1 else data.train_mask
    val_mask = data.val_mask[:, 0] if data.val_mask.dim() > 1 else data.val_mask
    test_mask = data.test_mask[:, 0] if data.test_mask.dim() > 1 else data.test_mask

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()
            
            if val_acc > best_test_acc: # Simplistic early stopping tracker
                best_test_acc = test_acc

    return best_test_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}\n")
    
    dataset_loaders = {
        'Chameleon': WikipediaNetwork(root="data/WikipediaNetwork", name="chameleon", transform=NormalizeFeatures()),
        'Texas': WebKB(root="data/WebKB", name="texas", transform=NormalizeFeatures()),
        'Roman-empire': HeterophilousGraphDataset(root="data/Heterophilous", name="Roman-empire", transform=NormalizeFeatures()),
        'Amazon-ratings': HeterophilousGraphDataset(root="data/Heterophilous", name="Amazon-ratings", transform=NormalizeFeatures()),
        'Minesweeper': HeterophilousGraphDataset(root="data/Heterophilous", name="Minesweeper", transform=NormalizeFeatures())
    }

    results = {ds: {} for ds in dataset_loaders.keys()}

    for ds_name, dataset in dataset_loaders.items():
        data = dataset[0].to(device)
        
        if ds_name in ['Chameleon', 'Texas']:
            data = deduplicate_graph(data)

        in_dim, out_dim = data.x.shape[1], dataset.num_classes
        
        models = {
            'GCN': GCN_Net(in_dim, 64, out_dim).to(device),
            'GAT': GAT_Net(in_dim, 64, out_dim).to(device),
            'H2GCN': H2GCN_Net(in_dim, 64, out_dim).to(device)
        }

        print(f"Running {ds_name} ({data.num_nodes} nodes)...")
        for model_name, model in models.items():
            acc = train_and_eval(model, data, device)
            results[ds_name][model_name] = acc * 100

    print("\n" + "="*55)
    print(f"{'Dataset':<15} | {'GCN (%)':<10} | {'GAT (%)':<10} | {'H2GCN (%)':<10}")
    print("-" * 55)
    for ds_name in dataset_loaders.keys():
        gcn, gat, h2gcn = results[ds_name]['GCN'], results[ds_name]['GAT'], results[ds_name]['H2GCN']
        print(f"{ds_name:<15} | {gcn:<10.2f} | {gat:<10.2f} | {h2gcn:<10.2f}")

if __name__ == "__main__":
    main()