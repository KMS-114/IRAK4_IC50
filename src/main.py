import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from torch_geometric.loader import DataLoader as GraphDataLoader

from dataloader import get_loaders, define_graphs, assign_features
from layers import Net
from utils import pIC50_to_IC50


def train_epoch(model: Net, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.
    iterator = tqdm(loader, desc="Training ... ")
    for idx, data in enumerate(iterator):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out.squeeze(-1), data.target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        iterator.set_postfix_str(f"loss={total_loss / (idx+1):.4}")
    return total_loss / len(loader)


@torch.no_grad()
def test_epoch(model: Net, loader):
    model.eval()
    all_tr = []
    all_pr = []
    iterator = tqdm(loader, desc="Testing ... ")
    for idx, data in enumerate(iterator):
        y_tr = data.target.cpu().numpy()
        y_pr = model(data).squeeze(-1).cpu().numpy()

        all_tr.append(y_tr)
        all_pr.append(y_pr)
    return np.concatenate(all_tr), np.concatenate(all_pr)


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.log(torch.cosh(y_pred - y_true)).mean()


if __name__ == '__main__':
    cfg = yaml.load(open("./configs.yml", "r"), Loader=yaml.FullLoader)

    graphs, fingerprints, targets, fields = define_graphs("train", **cfg["data"]["graph"])
    graphs = assign_features(graphs, fingerprints, fields, targets)
    trn_dl, tst_dl = get_loaders(graphs, **cfg["data"]["loader"])

    model = Net(fields=fields, **cfg["model"])
    optimizer = torch.optim.Adam(model.parameters(), **cfg["optimizer"])
    # loss_fn = nn.MSELoss()
    loss_fn = LogCoshLoss()

    best_score = -np.inf
    patience = 0
    for epoch in range(1, cfg["trainer"]["max_epochs"]+1):
        print(f"Epoch {epoch}")
        train_loss = train_epoch(model, trn_dl, optimizer, loss_fn)
        y_true, y_pred = test_epoch(model, tst_dl)

        correct_ratio = np.mean(np.abs(y_true - y_pred) <= 0.5)

        y_true = pIC50_to_IC50(y_true)
        y_pred = pIC50_to_IC50(y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        norm_rmse = rmse / (y_true.max() - y_true.min())

        score = (0.5 * min(1.-norm_rmse, 1.)) + (0.5 * correct_ratio)

        if best_score < score:
            best_score = score
            best_weights = model.state_dict()
            patience = 0
        else:
            patience += 1

        print(
            "\n###\n"
            f"norm_rmse={norm_rmse:.4f}, "
            f"min(1.-norm_rmse, 1)={min(1.-norm_rmse, 1.):.4f}, "
            f"correct_ratio={correct_ratio:.4f}, "
            f"score={score:.4f}, "
            f"best_score={best_score:.4f}, "
            f"patience={patience}"
            "\n###\n"
        )

        if cfg["trainer"]["early_stop"] <= patience:
            break

    model.load_state_dict(best_weights)
    model.eval()

    graphs, fingerprints = define_graphs("test", **cfg["data"]["graph"])
    graphs = assign_features(graphs, fingerprints, fields)
    test_dl = GraphDataLoader(graphs, batch_size=cfg["data"]["loader"]["test_batch_size"], shuffle=False)

    with torch.no_grad():
        all_pr = []
        iterator = tqdm(test_dl, desc="Testing ... ")
        for idx, data in enumerate(iterator):
            out = model(data)
            y_pr = pIC50_to_IC50(out.squeeze(-1).cpu().numpy())
            all_pr.append(y_pr)
        y_pred = np.concatenate(all_pr)

    submission = pd.read_csv("../data/sample_submission.csv")
    submission["IC50_nM"] = y_pred
    submission.to_csv("../data/submission.csv", index=False)