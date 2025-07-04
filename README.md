
# AugmenTox
## Transfer Learning Strategy for GNN Fine-Tuning on ToxCast Estrogen Receptor Data

This repository provides a graph neural network–based transfer learning pipeline to predict the activity of compounds on the Estrogen Receptor (ER) using the ToxCast dataset.

## Directory for execution
 ```bash
  cd model
  ```
## Requirements
- Python 3.9 or higher
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

## Code Structure
- `data/` – Contains data loaders that convert SMILES strings into graph representations and split them into training and evaluation sets
- `models/` – Implements GIN, GCN, GAT, and their hybrid variants
- `train/` – Training routines such as pretrain.py, finetune.py, and target_only.py
- `config/` – The config.py file where data paths and hyperparameters are defined
- `run_single_pipeline.sh` – A shell script that runs the full pretraining and fine-tuning pipeline for a single source/target combination
- `launcher.sh` – A Slurm launcher script to execute multiple combinations in parallel on an HPC cluster

## Quick Start

1. After installing dependencies, set the environment variables:
    ```bash
   export SOURCE_NAME=TOX21_ERa_LUC_VM7_Agonist export TARGET_NAME=ATG_ERE_CIS
    ```

2. Choose a model and run the pipeline. Supported models: GIN, GCN, GAT, GIN_GCN, GIN_GAT, GCN_GAT.
    ```bash
   python main.py --model GIN
    ```
    or (if you use Slurm)
   ```bash
    bash launcher.sh
   ```


When finished, the trained model will be saved under model/model_save/.

## Examples

### Running on a Single Combination
   ```bash
    export SOURCE_NAME=TOX21_ERa_LUC_VM7_Agonist export TARGET_NAME=ATG_ERE_CIS
    python main.py --model GIN
   ```


### Submitting an Array Job with Slurm
    bash launcher.sh


## License

This project is licensed under the terms of the [MIT License](LICENSE).

