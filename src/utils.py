import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.utils.convert import from_networkx


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


class FingerprintGenerator:
    def __init__(self, fp_size, radius):
        self.fp_size = fp_size
        self.fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)

    def smiles_to_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
             fp  = self.fp_gen.GetFingerprint(mol)
             return np.array(fp)
        else:
            return np.zeros((self.fp_size,))


def smiles_to_graph(smiles, fp_gen: FingerprintGenerator):
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

    # assign subgraph fingerprints
    sub_fps = {}
    for node in G.nodes():
        mol = Chem.RWMol()
        node_symbol = G.nodes("labels")[node]
        mol.AddAtom(Chem.Atom(node_symbol))

        for idx, neighbor in enumerate(G.neighbors(node)):
            neighbor_symbol = G.nodes("labels")[neighbor]
            mol.AddAtom(Chem.Atom(neighbor_symbol))
            mol.AddBond(0, idx+1, order=G.edges[node, neighbor]["bond_type"])

        sub_smiles = Chem.MolToSmiles(mol)
        sub_mol = Chem.MolFromSmiles(sub_smiles)
        if sub_mol is not None:
            fp = fp_gen.fp_gen.GetFingerprint(sub_mol)
            fp = np.array(fp)
        else:
            fp = np.zeros((fp_gen.fp_size, ))

        sub_fps[node] = {"sub_fp": fp}

    nx.set_node_attributes(G, sub_fps)
    return from_networkx(G)


def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)
