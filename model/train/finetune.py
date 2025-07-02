import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from train import utils
import config.config as cfg
from torch.utils.data import WeightedRandomSampler

def run_finetuning():
    # Load target dataset and split
    train_data, val_data, test_data = load_data(cfg.TARGET_DATA_PATH, 
                                                train_ratio=cfg.TRAIN_RATIO, 
                                                val_ratio=cfg.VAL_RATIO, 
                                                test_ratio=cfg.TEST_RATIO, 
                                                random_seed=cfg.RANDOM_SEED)
    input_dim = train_data[0].x.shape[1]
    output_dim = cfg.NUM_CLASSES
    # Select model architecture (same as pretrain)
    model_type = cfg.MODEL_TYPE.upper()
    if model_type == "GIN":
        from models.gin import GINNet
        model = GINNet(input_dim, cfg.HIDDEN_DIM, output_dim, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif model_type == "GCN":
        from models.gcn import GCNNet
        model = GCNNet(input_dim, cfg.HIDDEN_DIM, output_dim, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif model_type == "GAT":
        from models.gat import GATNet
        model = GATNet(input_dim, cfg.HIDDEN_DIM, output_dim, num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
    elif model_type == "GIN_GCN":
        from models.gin_gcn import GIN_GCN_Hybrid as ModelClass
        model = ModelClass(input_dim, cfg.HIDDEN_DIM, output_dim, num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif model_type == "GIN_GAT":
        from models.gin_gat import GIN_GAT_Hybrid as ModelClass
        model = ModelClass(input_dim, cfg.HIDDEN_DIM, output_dim, num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
    elif model_type == "GCN_GAT":
        from models.gcn_gat import GCN_GAT_Hybrid as ModelClass
        model = ModelClass(input_dim, cfg.HIDDEN_DIM, output_dim, num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
    else:
        raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=True)
    # Load pretrained weights
    utils.load_model(model, cfg.PRETRAINED_MODEL_PATH, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = utils.get_scheduler(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
    labels = [int(d.y.item()) for d in train_data]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=True)
    # Fine-tuning training loop
    best_val_f1 = 0.0
    best_state = None
    for epoch in range(1, cfg.NUM_EPOCHS_FINETUNE + 1):
        avg_loss = utils.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = utils.evaluate(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        scheduler.step()
        print(f"[Fine-tune] Epoch {epoch}/{cfg.NUM_EPOCHS_FINETUNE} - Loss: {avg_loss:.4f} - Val F1: {val_metrics['f1']:.4f} - Val ROC-AUC: {val_metrics['roc_auc']:.4f} - Val PR-AUC: {val_metrics['pr_auc']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    # Evaluate on target test set
    test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_metrics = utils.evaluate(model, test_loader, device)
    print(f"Finetuning completed. Test F1: {test_metrics['f1']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
    # Save finetuned model
    utils.save_model(model, cfg.FINETUNED_MODEL_PATH)

if __name__ == "__main__":
    run_finetuning()
