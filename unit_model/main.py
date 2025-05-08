import argparse
import config.config as cfg
from train import pretrain, finetune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN transfer learning experiment")
    parser.add_argument("--model", choices=["GIN", "GCN", "GAT"], default=cfg.MODEL_TYPE,
                        help="Model type to use for GNN (GIN, GCN, or GAT)")
    args = parser.parse_args()
    # Override model type in config if provided
    cfg.MODEL_TYPE = args.model
    print(f"Using model type: {cfg.MODEL_TYPE}")
    # Run pretraining on source dataset
    pretrain.run_pretraining()
    # Run finetuning on target dataset
    finetune.run_finetuning()
