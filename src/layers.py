import numpy as np
import torch
import torch.nn as nn

import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from torch_scatter import scatter_mean, scatter_max

from sklearn.preprocessing import LabelEncoder


def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0) 
    return torch.cat((x_mean, x_max), dim=-1)


class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims: dict[str, int], embed_dim=32):
        super().__init__()
        self.field_dims = field_dims
        self.embeddings = nn.ModuleDict({
            field_key: nn.Embedding(field_dim, embed_dim)
            for field_key, field_dim in field_dims.items()
        })
        self.lin = nn.Sequential(
            nn.Linear(len(field_dims)*embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: dict[str, torch.Tensor]):
        embs = []
        for key in self.field_dims.keys():
            tensor_ = torch.from_numpy(x[key]).to(x.edge_index.device)
            emb = self.embeddings[key](tensor_)
            embs.append(emb)

        embs = torch.cat(embs, dim=-1)
        x = self.lin(embs)
        return x


class  Net(nn.Module):
    def __init__(self,
                 fields: dict[str, LabelEncoder],
                 node_emb_dims,
                 hidden_dims,
                 num_layers,
                 fingerprint_dims):
        super(Net, self).__init__()
        self.fields = fields
        field_dims = {k: len(v.classes_) for k, v in fields.items()}
        self.projector = FieldAwareFactorizationMachine(field_dims=field_dims, embed_dim=node_emb_dims)
        self.fp_linear = nn.Linear(fingerprint_dims, hidden_dims)

        self.conv1 = gnn.GATConv(node_emb_dims, hidden_dims)
        self.pool1 = gnn.ASAPooling(hidden_dims)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(gnn.GATConv(hidden_dims, hidden_dims))
            self.pools.append(gnn.ASAPooling(hidden_dims))

        self.sub_fp_conv1 = gnn.GATConv(fingerprint_dims, hidden_dims)
        self.sub_fp_pool1 = gnn.ASAPooling(hidden_dims)
        self.sub_fp_convs = nn.ModuleList()
        self.sub_fp_pools = nn.ModuleList()
        for _ in range(num_layers-1):
            self.sub_fp_convs.append(gnn.GATConv(hidden_dims, hidden_dims))
            self.sub_fp_pools.append(gnn.ASAPooling(hidden_dims))

        self.lin1 = nn.Linear(hidden_dims*5, hidden_dims)
        self.lin2 = nn.Linear(hidden_dims, 1)

    def forward(self, data):
        bs = data.batch.max()+1

        x0 = [self.projector(data[i]) for i in range(bs)]
        x_cat = torch.cat(x0, dim=0)
        x_fp = torch.vstack([data[i]["sub_fp"] for i in range(bs)]).to(torch.float32)
        # x_fp = torch.from_numpy(x_fp).to(torch.float32)

        # forward for Node Categorical features
        x_cat = self.conv1(x_cat, data.edge_index).relu()
        x_cat, edge_index, _, batch, _ = self.pool1(x_cat, data.edge_index, batch=data.batch)
        xs_cat = readout(x_cat, batch)
        for conv, pool in zip(self.convs, self.pools):
            x_cat = conv(x_cat, edge_index).relu()
            x_cat, edge_index, _, batch, _ = pool(x_cat, edge_index, batch=batch)
            xs_cat += readout(x_cat, batch)

        # forward for Sub Fingerpring features
        x_fp = self.sub_fp_conv1(x_fp, data.edge_index).relu()
        x_fp, edge_index, _, batch, _ = self.sub_fp_pool1(x_fp, data.edge_index, batch=data.batch)
        xs_fp = readout(x_fp, batch)
        for conv, pool in zip(self.sub_fp_convs, self.sub_fp_pools):
            x_fp = conv(x_fp, edge_index).relu()
            x_fp, edge_index, _, batch, _ = pool(x_fp, edge_index, batch=batch)
            xs_fp += readout(x_fp, batch)

        # x_cat = scatter(src=x_cat, index=data.batch, dim=0)
        # x_fp = scatter(src=x_fp, index=data.batch, dim=0)

        fp_emb = torch.from_numpy(np.stack(data.fingerprint)).to(x_fp)
        fp_emb = self.fp_linear(fp_emb)

        x = torch.cat([fp_emb, xs_cat, xs_fp], dim=-1)

        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x