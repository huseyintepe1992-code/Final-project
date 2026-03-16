import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.sparse as sp
from torch_geometric.datasets import WikipediaNetwork, WebKB
from torch_geometric.transforms import NormalizeFeatures

def generate_pca_plots():
    datasets_info = [
        ('Chameleon', WikipediaNetwork(root="data/WikipediaNetwork", name="chameleon", transform=NormalizeFeatures())),
        ('Texas', WebKB(root="data/WebKB", name="texas", transform=NormalizeFeatures()))
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PCA Analysis of Heterophilic Graphs: Chameleon & Texas", fontsize=16, fontweight='bold', y=1.02)

    for i, (name, dataset) in enumerate(datasets_info):
        data = dataset[0]
        x = data.x.numpy()
        y = data.y.numpy()
        row, col = data.edge_index.numpy()
        N = data.num_nodes

        # Adjacency and Laplacian
        A = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(N, N)).maximum(sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(N, N)).T)
        deg = np.array(A.sum(axis=1)).flatten()
        D_inv = sp.diags(1.0 / (deg + 1e-10))
        A_norm = D_inv @ A

        # 1. Raw PCA
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x)
        axes[i, 0].scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
        axes[i, 0].set_title(f"{name}: Raw Node Features")
        axes[i, 0].set_ylabel(name, fontsize=12, fontweight='bold')
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        # 2. Low-Pass (Navier-Stokes style diffusion)
        x_diff = A_norm @ x
        x_diff_pca = pca.fit_transform(x_diff)
        axes[i, 1].scatter(x_diff_pca[:, 0], x_diff_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
        axes[i, 1].set_title(f"{name}: Low-Pass (Smoothed)")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

        # 3. High-Pass (Laplacian / Discrete state separation)
        x_lap = x - x_diff
        x_lap_pca = pca.fit_transform(x_lap)
        axes[i, 2].scatter(x_lap_pca[:, 0], x_lap_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
        axes[i, 2].set_title(f"{name}: High-Pass (Laplacian)")
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

    plt.tight_layout()
    plt.savefig('chameleon_texas_pca.png', dpi=150, bbox_inches='tight')
    print("Saved PCA visualization as 'chameleon_texas_pca.png'")

if __name__ == "__main__":
    generate_pca_plots()