import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_embeddings(model, data_list, device, save_path=None):
    """
    Compute graph embeddings for each graph in data_list using the model, 
    reduce to 2D with t-SNE, and plot a scatter of the embeddings colored by their labels.
    """
    model.eval()
    model.to(device, non_blocking=True)
    embeddings = []
    labels = []
    # Compute embeddings
    with torch.no_grad():
        for data in data_list:
            # batch 정보가 없으면 모든 노드를 그래프 0번으로 처리
            if not hasattr(data, 'batch') or data.batch is None:
                num_nodes = data.x.size(0)
                data.batch = torch.zeros(num_nodes, dtype=torch.long)
            data = data.to(device)
            emb = model.get_graph_embedding(data)
            # emb is shape [hidden_dim] for one graph
            embeddings.append(emb.cpu().numpy())
            # assume binary classification (single label in data.y)
            label = int(data.y.item()) if data.y.dim() == 1 else int(data.y.argmax().item())
            labels.append(label)
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    # 임베딩 차원이 2가 아니면 TSNE로 2차원으로 변환
    if embeddings.shape[1] != 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings  # 이미 2차원인 경우는 그대로 사용

    # Plot the embeddings
    plt.figure(figsize=(6,5))
    labels = np.array(labels)
    if len(np.unique(labels)) == 2:
        # binary classification
        plt.scatter(embeddings_2d[labels==0, 0], embeddings_2d[labels==0, 1], c='tab:blue', label='Class 0', alpha=0.7)
        plt.scatter(embeddings_2d[labels==1, 0], embeddings_2d[labels==1, 1], c='tab:orange', label='Class 1', alpha=0.7)
    else:
        # multi-class or regression (treat each unique label with a color map)
        cmap = plt.get_cmap("tab10")
        for lab in np.unique(labels):
            plt.scatter(embeddings_2d[labels==lab, 0], embeddings_2d[labels==lab, 1],
                        label=f"Class {lab}", alpha=0.7)
    plt.legend()
    plt.title("t-SNE Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
