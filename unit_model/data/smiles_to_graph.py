from rdkit import Chem
import torch
from torch_geometric.data import Data

def one_hot_encoding(x, allowed_values):
    """
    Maps input x to a one-hot encoding based on a list of allowed values.
    If x is not in allowed_values, it will be mapped to the last element's position.
    """
    if x not in allowed_values:
        x = allowed_values[-1]
    return [1 if x == v else 0 for v in allowed_values]

def get_atom_features(atom):
    """
    Computes a feature vector for a single atom.
    Features:
    - Atom type (one-hot): common atoms and one 'Unknown'
    - Atom degree (one-hot up to 4, then 'MoreThan4')
    - Formal charge (one-hot for -3 to +3, 'Extreme' for others)
    - Aromaticity (1 if aromatic, else 0)
    """
    # Define possible features
    atom_symbol = atom.GetSymbol()
    # List of possible atom symbols (common organic atoms + "Unknown")
    permitted_atoms = ['H','C','N','O','F','P','S','Cl','Br','I','Unknown']
    atom_type_enc = one_hot_encoding(atom_symbol, permitted_atoms)

    # Atom degree (number of neighbors)
    atom_degree = atom.GetDegree()
    permitted_degrees = [0, 1, 2, 3, 4, "MoreThan4"]
    if atom_degree >= 5:
        atom_degree = "MoreThan4"
    degree_enc = one_hot_encoding(atom_degree, permitted_degrees)

    # Formal charge
    formal_charge = atom.GetFormalCharge()
    permitted_charges = [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    if formal_charge < -3 or formal_charge > 3:
        formal_charge = "Extreme"
    charge_enc = one_hot_encoding(formal_charge, permitted_charges)

    # Aromaticity
    aromatic_enc = [1 if atom.GetIsAromatic() else 0]

    # Combine all features
    atom_feature_vector = atom_type_enc + degree_enc + charge_enc + aromatic_enc
    return atom_feature_vector

def get_bond_features(bond):
    """
    Computes a feature vector for a single bond.
    Features:
    - Bond type (one-hot for single, double, triple, aromatic)
    - Conjugation (1 if bond is conjugated, else 0)
    - InRing (1 if bond is in a ring, else 0)
    """
    # Possible bond types
    bt = bond.GetBondType()
    permitted_bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bt, permitted_bond_types)
    # Conjugation and ring
    bond_conj_enc = [1 if bond.GetIsConjugated() else 0]
    bond_ring_enc = [1 if bond.IsInRing() else 0]
    bond_feature_vector = bond_type_enc + bond_conj_enc + bond_ring_enc
    return bond_feature_vector

def smiles_to_graph(smiles):
    """
    Converts a SMILES string to a PyTorch Geometric Data object (graph).
    Nodes represent atoms with features, edges represent bonds with features.
    Returns a Data object with x (node features), edge_index, edge_attr, and empty y (label to be assigned).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return an empty graph if SMILES is invalid
        return Data()
    # Node features
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    # Edges: we add edges in both directions for an undirected graph representation
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # edge from i to j
        edge_indices.append([i, j])
        edge_features.append(get_bond_features(bond))
        # edge from j to i
        edge_indices.append([j, i])
        edge_features.append(get_bond_features(bond))
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Molecule with no bonds (like single atom molecule)
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_bond_features(Chem.MolFromSmiles('O=O').GetBondWithIdx(0)))), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
