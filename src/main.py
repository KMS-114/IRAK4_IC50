import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import smiles_to_fingerprint, smiles_to_graph


COLUMNS = [
    "Smiles", "pIC50"
]


if __name__ == '__main__':
    chembl_data = pd.read_csv('../data/train.csv')
    df = chembl_data[COLUMNS]
    df["fingerprint"] = df["Smiles"].apply(smiles_to_fingerprint)

    graph_0 = smiles_to_graph(df.loc[0, "Smiles"])
    df["graph"] = df["Smiles"].apply(smiles_to_graph)

    features = np.stack(df["fingerprint"].values)
    target = df["pIC50"].values

    train_x, val_x, train_y, val_y = train_test_split(features, target, test_size=0.3, random_state=42)
