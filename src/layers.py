import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

from sklearn.preprocessing import LabelEncoder


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
            tensor_ = torch.from_numpy(x[key])
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

        self.convs = nn.ModuleList([
            GCNConv(node_emb_dims, hidden_dims),
            *[GCNConv(hidden_dims, hidden_dims) for i in range(num_layers-1)]
        ])

        self.lin1 = nn.Linear(hidden_dims*2, hidden_dims//2)
        self.lin2 = nn.Linear(hidden_dims//2, 1)
    
    def forward(self, data):
        bs = data.batch.max()+1

        x0 = [self.projector(data[i]) for i in range(bs)]
        x = torch.cat(x0, dim=0)

        for conv in self.convs:
            x = conv(x, data.edge_index).relu()

        x = scatter(src=x, index=data.batch, dim=0)

        fp_emb = torch.from_numpy(np.stack(data.fingerprint)).to(dtype=torch.float32)
        fp_emb = self.fp_linear(fp_emb)
        x = torch.cat([fp_emb, x], dim=-1)

        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x