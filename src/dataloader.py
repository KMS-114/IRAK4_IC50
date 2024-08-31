import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch_geometric.loader import DataLoader as GraphDataLoader

from utils import smiles_to_graph, FingerprintGenerator


COLUMNS = [
    "Smiles", "pIC50"
]


def get_loaders(graphs, train_batch_size, test_batch_size):
    # train-validation split
    train_indices, test_indices = train_test_split(np.arange(len(graphs)), test_size=0.3, random_state=42)
    trn_graphs = graphs[train_indices]
    tst_graphs = graphs[test_indices]

    trn_dl = GraphDataLoader(trn_graphs, batch_size=train_batch_size, shuffle=True)
    tst_dl = GraphDataLoader(tst_graphs, batch_size=test_batch_size, shuffle=False)
    return trn_dl, tst_dl


def define_graphs(split, data_source, field_keys, fingerprint_dims, radius):
    # Load data
    df = pd.read_csv(os.path.join(data_source, f"{split}.csv"))

    # Get fingerprint features
    fp_gen = FingerprintGenerator(fingerprint_dims, radius)
    df.loc[:, "fingerprint"] = df["Smiles"].apply(fp_gen.smiles_to_fingerprint)

    fingerprints = np.stack(df["fingerprint"].values)

    # Get graph datasets from SMILES codes
    graphs = df["Smiles"].apply(smiles_to_graph, args=(fp_gen, )).values

    if split == "train":
        target = df["pIC50"].values
        # Define Atomic feature encoder
        fields = {key: [] for key in field_keys}
        for g in graphs:
            for k, v in fields.items():
                v += g[k]

        for key in field_keys:
            encoder = LabelEncoder()
            encoder.fit(fields[key])
            fields[key] = encoder
        return graphs, fingerprints, target, fields
    else:
        return graphs, fingerprints


def assign_features(graphs, fingerprints, fields, targets=None):
    for g, fp in zip(graphs, fingerprints):
        g = g.update({"fingerprint": fp})
        for field_key, encoder in fields.items():
            g = g.update({field_key: encoder.transform(g[field_key])})

    if targets is not None:
        for g, t in zip(graphs, targets):
            g = g.update({"target": t})
    return graphs