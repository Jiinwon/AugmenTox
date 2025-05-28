# train/target_only.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from train.utils import train_one_epoch, evaluate, save_model, get_scheduler
import config.config as cfg
from torch.utils.data import WeightedRandomSampler

def run_target_only():
    # 1) load and split the target dataset
    train_data, val_data, test_data = load_data(
        cfg.TARGET_DATA_PATH,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        random_seed=cfg.RANDOM_SEED
    )

  # 2) build the model
    input_dim = train_data[0].x.shape[1]
    output_dim = cfg.NUM_CLASSES
    model_type = cfg.MODEL_TYPE.upper()

    # 기본 파라미터
    model_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": cfg.HIDDEN_DIM,
        "output_dim": output_dim,
        "num_layers": cfg.NUM_LAYERS,
        "dropout": cfg.DROPOUT,
    }

    # 모델 클래스 선택
    if model_type == "GIN":
        from models.gin import GINNet as Net
    elif model_type == "GCN":
        from models.gcn import GCNNet as Net
    elif model_type == "GAT":
        from models.gat import GATNet as Net
        model_kwargs["heads"] = cfg.NUM_HEADS
    elif model_type == "GIN_GCN":
        from models.gin_gcn import GIN_GCN_Hybrid as Net
    elif model_type == "GIN_GAT":
        from models.gin_gat import GIN_GAT_Hybrid as Net
        model_kwargs["heads"] = cfg.NUM_HEADS
    elif model_type == "GCN_GAT":
        from models.gcn_gat import GCN_GAT_Hybrid as Net
        model_kwargs["heads"] = cfg.NUM_HEADS
    else:
        raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")

    # 모델 생성 (공통)
    model = Net(**model_kwargs)
    

    # 3) training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = get_scheduler(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

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

    val_loader   = DataLoader(val_data,   batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 4) train
    best_val_f1 = 0.0
    best_state = None
    print("=== Target-only training ===")
    for epoch in range(1, cfg.NUM_EPOCHS_FINETUNE+1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
        scheduler.step()
        print(f"[Target-only] Epoch {epoch}/{cfg.NUM_EPOCHS_FINETUNE} – "
              f"Loss: {loss:.4f}  Val F1: {metrics['f1']:.4f}  "
              f"ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # 5) final test
    test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_metrics = evaluate(model, test_loader, device)
    print(f"=== Target-only TEST performance ===")
    print(f"Test F1: {test_metrics['f1']:.4f}, "
          f"ROC-AUC: {test_metrics['roc_auc']:.4f}, "
          f"PR-AUC: {test_metrics['pr_auc']:.4f}")

    # 6) optionally save
    save_model(model, f"model_save/target_only_{cfg.MODEL_TYPE.lower()}.pth")


if __name__ == "__main__":
    run_target_only()