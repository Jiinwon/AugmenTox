import os
import torch
import torch.nn.functional as F
from eval import metrics as metrics

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # If binary classification, out shape [batch, 1] and data.y shape [batch], or [batch,1].
        # Ensure shapes align
        logits = out.view(-1)
        target = data.y.view(-1).to(device)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Evaluate model on data_loader, return metrics (F1, precision, recall, ROC-AUC, PR-AUC)."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(data.y.view(-1).long().cpu().tolist())
    # Compute metrics using sklearn functions
    f1 = metrics.get_f1(all_labels, all_preds)
    precision = metrics.get_precision(all_labels, all_preds)
    recall = metrics.get_recall(all_labels, all_preds)
    roc_auc = metrics.get_roc_auc(all_labels, all_probs)
    pr_auc = metrics.get_pr_auc(all_labels, all_probs)
    results = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
    return results

def save_model(model, path):
    """Save the model weights to the given file path.
    Creates directories if they do not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load model weights from the given file path."""
    model.load_state_dict(torch.load(path, map_location=device))

def get_scheduler(optimizer, step_size=10, gamma=0.5):
    """Return a StepLR scheduler."""
    from torch.optim.lr_scheduler import StepLR
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
