import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return [int(x == s) for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    # Maps inputs not in allowable_set to the last element
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom):
    # One-hot for atom symbol
    symbols = [
        'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al',
        'I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H',
        'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
    ]
    symbol_enc = one_of_k_encoding_unk(atom.GetSymbol(), symbols)

    # Atom degree (0-10)
    degree_enc = one_of_k_encoding(atom.GetDegree(), list(range(11)))

    # Total number of Hs (0-10)
    total_hs_enc = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))

    # Implicit valence (0-10)
    implicit_valence_enc = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))

    # Aromaticity
    aromatic_enc = [int(atom.GetIsAromatic())]

    features = symbol_enc + degree_enc + total_hs_enc + implicit_valence_enc + aromatic_enc
    # Convert to numpy array and normalize
    arr = np.array(features, dtype=float)
    if arr.sum() > 0:
        arr = arr / arr.sum()
    return arr


def smiles_to_graph(smiles: str) -> Data:
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    Node features include atom_features; edge_index is directed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return empty graph
        return Data(x=torch.empty((0, 0)), edge_index=torch.empty((2, 0), dtype=torch.long))

    # Node features
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.stack(features), dtype=torch.float)

    # Edges (undirected -> directed)
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append((i, j))
        edge_list.append((j, i))
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Molecule with no bonds (e.g., single atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data
