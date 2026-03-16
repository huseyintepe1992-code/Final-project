import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.sparse as sp
import os

datasets = {
    'Roman-empire': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/roman_empire.npz',
    'Amazon-ratings': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/amazon_ratings.npz',
    'Minesweeper': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/minesweeper.npz'
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("PCA Analysis of Heterophilic Graphs: Roman-empire, Amazon-rating & Minesweeper", fontsize=16, fontweight='bold', y=1.02)

for i, (name, url) in enumerate(datasets.items()):
    # Download
    filepath = f"{name.lower()}.npz"
    if not os.path.exists(filepath):
        try:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            continue
            
    # Load
    data = np.load(filepath)
    x = data['node_features']
    y = data['node_labels']
    edges = data['edges'] 
    
    # Subsample for PCA clarity and speed if dataset is huge
    np.random.seed(42)
    sample_size = min(3000, len(y))
    idx = np.random.choice(len(y), sample_size, replace=False)
        
    x_samp = x[idx]
    y_samp = y[idx]
    
    # --- 1. Raw PCA ---
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_samp)
    
    ax = axes[i, 0]
    ax.scatter(x_pca[:, 0], x_pca[:, 1], c=y_samp, cmap='tab10', alpha=0.05, s=20)
    ax.set_title(f"{name}: Raw Node Features")
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 2: ax.set_xlabel("PCA 1")
    if i == 0: ax.set_ylabel("Roman-Empire", fontsize=12, fontweight='bold')
    if i == 1: ax.set_ylabel("Amazon-Ratings", fontsize=12, fontweight='bold')
    if i == 2: ax.set_ylabel("Minesweeper", fontsize=12, fontweight='bold')
    
    # --- Simulate Navier-Stokes Diffusion (Positive Viscosity) ---
    N = len(y)
    row = edges[:, 0]
    col = edges[:, 1]
    
    # Create Adjacency Matrix
    A = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(N, N))
    # Symmetrize
    A = A.maximum(A.T)
    
    # Normalize A (Mean aggregation)
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = 1.0 / (deg + 1e-10)
    D_inv = sp.diags(deg_inv)
    A_norm = D_inv @ A
    
    # One step of Navier-Stokes positive diffusion (smoothing)
    x_diff = A_norm @ x 
    x_diff_samp = x_diff[idx]
    x_diff_pca = pca.fit_transform(x_diff_samp)
    
    ax = axes[i, 1]
    ax.scatter(x_diff_pca[:, 0], x_diff_pca[:, 1], c=y_samp, cmap='tab10', alpha=0.5, s=8)
    ax.set_title(f"{name}: Sim. Navier-Stokes (Low-Pass)")
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 2: ax.set_xlabel("PCA 1")
    
    # --- Simulate High-Pass Filter (Laplacian / Negative Viscosity) ---
    # L = I - A_norm
    x_lap = x - x_diff
    x_lap_samp = x_lap[idx]
    x_lap_pca = pca.fit_transform(x_lap_samp)
    
    ax = axes[i, 2]
    ax.scatter(x_lap_pca[:, 0], x_lap_pca[:, 1], c=y_samp, cmap='tab10', alpha=0.5, s=8)
    ax.set_title(f"{name}: Laplacian Filter (High-Pass)")
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 2: ax.set_xlabel("PCA 1")

plt.tight_layout()
plt.savefig('heterophily_pca_analysis.png', dpi=150, bbox_inches='tight')
print("Successfully generated PCA visualization.")