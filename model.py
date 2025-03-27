import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class ProteinGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, mode="binary", use_sas_weight=True):
        super(ProteinGAT, self).__init__()

        self.mode = mode.lower()  # "binary" or "multiclass"
        self.use_sas_weight = use_sas_weight

        edge_dim = 3 if use_sas_weight else 2

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)

        out = self.fc(x)  # shape: [n_nodes, output_dim]

        # if self.mode == "binary":
        #     return torch.sigmoid(out).squeeze(-1)  # shape: [n_nodes]
        # else:
        #     return F.log_softmax(out, dim=-1)  # shape: [n_nodes, num_classes]
        return out.squeeze(-1)
    