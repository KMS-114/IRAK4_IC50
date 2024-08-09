import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem


fp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        # return np.array(fp)
        fp = fp_gen.GetFingerprint(mol)
        return np.array(fp)
    else:
        return np.zeros((1024,))


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            labels=atom.GetSymbol(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag(),
            hybridization=atom.GetHybridization(),
            num_explicit_hs=atom.GetNumExplicitHs(),
            is_aromatic=atom.GetIsAromatic(),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType(),
            bond_type_value=bond.GetBondTypeAsDouble(),
        )
    return G


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)
