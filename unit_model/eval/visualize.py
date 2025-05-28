import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import os
import config.config as cfg

def visualize_embeddings(model, data_list, device, save_path=None):
    """
    Compute graph embeddings for each graph in data_list using the model, 
    reduce to 2D with t-SNE, and plot a scatter of the embeddings colored by their labels.
    """
    model.eval()
    model.to(device, non_blocking=True)
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in data_list:
            if not hasattr(data, 'batch') or data.batch is None:
                num_nodes = data.x.size(0)
                data.batch = torch.zeros(num_nodes, dtype=torch.long)
            data = data.to(device)
            emb = model.get_graph_embedding(data)
            embeddings.append(emb.cpu().numpy())
            label = int(data.y.item()) if data.y.dim() == 1 else int(data.y.argmax().item())
            labels.append(label)

    embeddings = np.array(embeddings).reshape(len(embeddings), -1)

    if embeddings.shape[1] != 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    plt.figure(figsize=(6, 5))
    labels = np.array(labels)
    if len(np.unique(labels)) == 2:
        plt.scatter(embeddings_2d[labels==0, 0], embeddings_2d[labels==0, 1], c='tab:blue', label='Class 0', alpha=0.7)
        plt.scatter(embeddings_2d[labels==1, 0], embeddings_2d[labels==1, 1], c='tab:orange', label='Class 1', alpha=0.7)
    else:
        cmap = plt.get_cmap("tab10")
        for lab in np.unique(labels):
            plt.scatter(embeddings_2d[labels==lab, 0], embeddings_2d[labels==lab, 1],
                        label=f"Class {lab}", alpha=0.7)
    plt.legend()
    plt.title("t-SNE Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # 저장 디렉토리 설정
    if save_path is None:
        save_dir = os.path.join("unit_model", "figure")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"figure/embedding_tsne_{cfg.TARGET_NAME}&&{cfg.SOURCE_NAME}_{cfg.MODEL_NAME}.png"
        save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path)
    plt.close()
