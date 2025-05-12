def load_data(csv_path, smiles_col='smiles', label_col='label', 
              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from collections import Counter
    from data.smiles_to_graph import smiles_to_graph
    
    # if someone wants 100% as test, just return everything as test
    if train_ratio == 0 and val_ratio == 0 and test_ratio == 1:
        df = pd.read_excel(file_path) if file_path.endswith(('.xls','xlsx')) else pd.read_csv(file_path)
        df = df.dropna(subset=[smiles_col, label_col])
        data_list = []
        for smi, lbl in zip(df[smiles_col].astype(str), df[label_col]):
            data = smiles_to_graph(smi)
            if data is None or not hasattr(data,'x') or data.x.size(0)==0:
                continue
            data.y = torch.tensor([float(lbl)], dtype=torch.float)
            data_list.append(data)
        return [], [], data_list
    
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{smiles_col}' and '{label_col}'.")

    smiles_list = df[smiles_col].astype(str).tolist()
    labels_list = df[label_col].tolist()

    data_list = []
    for smi, label in zip(smiles_list, labels_list):
        data = smiles_to_graph(smi)
        if data is None or not hasattr(data, 'x') or data.x is None or data.x.size(0) == 0:
            continue
        data.y = torch.tensor([float(label)], dtype=torch.float)
        data_list.append(data)

    labels_arr = [int(float(data.y)) for data in data_list]
    class_counts = Counter(labels_arr)

    if len(class_counts) < 2:
        print(f"[WARNING] Only one class present ({class_counts}). Returning all as train_data.")
        return data_list, [], []

    # 정상적으로 stratified split 진행
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data_list, labels_arr, test_size=(val_ratio + test_ratio),
        stratify=labels_arr, random_state=random_seed)

    rel_test_ratio = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
    if rel_test_ratio > 0:
        val_data, test_data, _, _ = train_test_split(
            temp_data, temp_labels, test_size=rel_test_ratio,
            stratify=temp_labels, random_state=random_seed)
    else:
        val_data = []
        test_data = temp_data

    return train_data, val_data, test_data
