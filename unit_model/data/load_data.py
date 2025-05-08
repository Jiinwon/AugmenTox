import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data.smiles_to_graph import smiles_to_graph

def load_data(csv_path, smiles_col='smiles', label_col='label', 
              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Loads a CSV file containing SMILES strings and labels, converts each SMILES to a graph Data object,
    and splits the dataset into training, validation, and test sets.
    Returns a tuple: (train_data_list, val_data_list, test_data_list)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{smiles_col}' and '{label_col}'.")
    smiles_list = df[smiles_col].astype(str).tolist()
    labels_list = df[label_col].tolist()
    # Convert SMILES to graph Data objects
    data_list = []
    for smi, label in zip(smiles_list, labels_list):
        data = smiles_to_graph(smi)
        # Attach label (binary classification) to Data object
        data.y = torch.tensor([float(label)], dtype=torch.float)
        data_list.append(data)
    # Stratified split into train, val, test
    # Prepare array of labels for stratification (0/1 classes)
    labels_arr = [int(float(data.y)) for data in data_list]
    # First split: train vs temp (val+test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data_list, labels_arr, test_size=(val_ratio + test_ratio),
        stratify=labels_arr, random_state=random_seed)
    # Second split: val vs test from temp
    if val_ratio + test_ratio > 0:
        # Compute relative fraction for test portion of temp
        rel_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_data, test_data, _, _ = train_test_split(
            temp_data, temp_labels, test_size=rel_test_ratio,
            stratify=temp_labels, random_state=random_seed)
    else:
        val_data = []
        test_data = temp_data
    return train_data, val_data, test_data
