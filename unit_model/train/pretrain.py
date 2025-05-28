import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from train import utils
import config.config as cfg
from torch.utils.data import WeightedRandomSampler

def run_pretraining():
    # Load source dataset and split
    train_data, val_data, test_data = load_data(cfg.SOURCE_DATA_PATH, 
                                                train_ratio=cfg.TRAIN_RATIO, 
                                                val_ratio=cfg.VAL_RATIO, 
                                                test_ratio=cfg.TEST_RATIO, 
                                                random_seed=cfg.RANDOM_SEED)
    # Determine input dimension from data
    input_dim = train_data[0].x.shape[1]
    # CSV 모드일 땐 1, SDF 모드(OPERA)일 땐 ENDPOINTS 길이
    if cfg.OPERA:
        output_dim = len(cfg.ENDPOINTS)
    else:
        output_dim = 1
    # Select model architecture
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

    # Prepare for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = utils.get_scheduler(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
    # 레이블 리스트 추출
    # CSV 모드: 스칼라, SDF 모드: vector → primary label (예: 첫 번째)로 샘플링
    labels = []
    if cfg.OPERA:
        # 첫 번째 ENDPOINTS 값에 매핑되는 인덱스 찾기 (없으면 0으로 fallback)
        primary_field = cfg.ENDPOINTS[0]
        if primary_field in cfg.SDF_LABEL_FIELDS:
            primary_idx = cfg.SDF_LABEL_FIELDS.index(primary_field)
        else:
            primary_idx = 0

        for d in train_data:
            y = d.y
            # 벡터 길이보다 primary_idx가 크면 0번 요소로 대체
            idx = primary_idx if y.numel() > primary_idx else 0
            labels.append(int(y[idx].item()))
    else:
        for d in train_data:
            labels.append(int(d.y.item()))
    # 클래스별 샘플 수
    class_counts = [labels.count(0), labels.count(1)]
    # 각 샘플에 대해 inverse frequency 가중치 부여
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
    # Training loop
    best_val_f1 = 0.0
    best_state = None
    for epoch in range(1, cfg.NUM_EPOCHS_PRETRAIN + 1):
        avg_loss = utils.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = utils.evaluate(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        # Step the scheduler
        scheduler.step()
        print(f"Epoch {epoch}/{cfg.NUM_EPOCHS_PRETRAIN} - Loss: {avg_loss:.4f} - Val F1: {val_metrics['f1']:.4f} - Val ROC-AUC: {val_metrics['roc_auc']:.4f} - Val PR-AUC: {val_metrics['pr_auc']:.4f}")
    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    # Evaluate on source test set
    test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_metrics = utils.evaluate(model, test_loader, device)
    print(f"Pretraining completed. Test F1: {test_metrics['f1']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
    # Save the pretrained model
    utils.save_model(model, cfg.PRETRAINED_MODEL_PATH)

if __name__ == "__main__":
    run_pretraining()
