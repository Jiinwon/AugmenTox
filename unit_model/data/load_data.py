def load_data(path, smiles_col='smiles', label_col='label',
              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from collections import Counter
    from data.smiles_to_graph import smiles_to_graph
    from rdkit import Chem
    import config.config as cfg

    # --- SDF 파일 입력 처리 (멀티-레이블) ---
    if path.lower().endswith('.sdf'):
        suppl = Chem.SDMolSupplier(path)
        data_list = []
        # SDF에서 사용할 클래스 필드 이름
        # ① 전체 가능한 필드와, ② config.ENDPOINTS 에 맞춰 사용할 필드만 고른다
        all_fields = cfg.SDF_LABEL_FIELDS
        if "all" in cfg.ENDPOINTS:
            sdf_label_fields = all_fields
        else:
            # ENDPOINTS 에 지정된 이름이 반드시 all_fields에 있어야 합니다
            sdf_label_fields = [f for f in all_fields if f in cfg.ENDPOINTS]
        if not sdf_label_fields:
            raise ValueError(f"No matching SDF label fields for ENDPOINTS={cfg.ENDPOINTS}")

        for mol in suppl:
            if mol is None:
                continue

            # 선택된 필드만 읽어서 labels 벡터 생성
            labels = [ float(mol.GetProp(f)) for f in sdf_label_fields if mol.HasProp(f) ]
            if len(labels) != len(sdf_label_fields):
                continue

            data = smiles_to_graph(Chem.MolToSmiles(mol))
            if data is None or not hasattr(data, 'x') or data.x.size(0) == 0:
                continue

            data.y = torch.tensor(labels, dtype=torch.float)  # shape = [len(sdf_label_fields)]
            data_list.append(data)

        # 레이블 분포 체크
        label_ints = [int(v) for data in data_list for v in data.y.tolist()]
        if len(set(label_ints)) < 2:
            return data_list, [], []

        # Stratified split (단일 벡터는 불가능하므로, 여기서는 binding 클래스만 기준으로 예시)
        # 필요하다면 다중 레이블 stratify 패키지를 사용하세요.
        # Stratify 기준: config.ENDPOINTS 순서대로 결정된 sdf_label_fields[0]만 사용
        # (벡터 길이에 상관없이 index 0 이 항상 유효)
        stratify_idx = 0
        primary_labels = [
            int(data.y[stratify_idx].item())
            for data in data_list
        ]
        train_data, temp_data, train_lbl, temp_lbl = train_test_split(
            data_list, primary_labels,
            test_size=(val_ratio + test_ratio),
            stratify=primary_labels,
            random_state=random_seed
        )
        rel_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
        if rel_test > 0:
            val_data, test_data, _, _ = train_test_split(
                temp_data, temp_lbl,
                test_size=rel_test,
                stratify=temp_lbl,
                random_state=random_seed
            )
        else:
            val_data, test_data = [], temp_data

        return train_data, val_data, test_data
    # --- SDF 분기 끝 ---

    # --- 기존 CSV 입력 처리 (변경 없음) ---
    # 100% test 모드
    if train_ratio == 0 and val_ratio == 0 and test_ratio == 1:
        df = pd.read_excel(path) if path.endswith(('.xls','xlsx')) else pd.read_csv(path)
        df = df.dropna(subset=[smiles_col, label_col])
        data_list = []
        for smi, lbl in zip(df[smiles_col].astype(str), df[label_col]):
            data = smiles_to_graph(smi)
            if data is None or not hasattr(data, 'x') or data.x.size(0) == 0:
                continue
            data.y = torch.tensor([float(lbl)], dtype=torch.float)
            data_list.append(data)
        return [], [], data_list

    # 일반 CSV 분할 모드
    df = pd.read_csv(path)
    if smiles_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{smiles_col}' and '{label_col}'.")

    data_list = []
    for smi, lbl in zip(df[smiles_col].astype(str), df[label_col]):
        data = smiles_to_graph(smi)
        if data is None or not hasattr(data, 'x') or data.x.size(0) == 0:
            continue
        data.y = torch.tensor([float(lbl)], dtype=torch.float)
        data_list.append(data)

    labels_arr = [int(data.y.item()) for data in data_list]
    class_counts = Counter(labels_arr)
    if len(class_counts) < 2:
        return data_list, [], []

    train_data, temp_data, train_lbls, temp_lbls = train_test_split(
        data_list, labels_arr,
        test_size=(val_ratio + test_ratio),
        stratify=labels_arr,
        random_state=random_seed
    )
    rel_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
    if rel_test > 0:
        val_data, test_data, _, _ = train_test_split(
            temp_data, temp_lbls,
            test_size=rel_test,
            stratify=temp_lbls,
            random_state=random_seed
        )
    else:
        val_data, test_data = [], temp_data

    return train_data, val_data, test_data