import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv


################################################################################
# MODELS
################################################################################

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.lin2 = torch.nn.Linear(hidden_dim//2, embedding_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = self.lin2(x)
        return embedding

