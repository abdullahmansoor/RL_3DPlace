import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader


################################################################################
# MODELS
################################################################################

# Encoder: Generates embeddings from graph structures
class GINEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding_dim):
        super(GINEncoder, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        self.convs = torch.nn.ModuleList([
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Sigmoid(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )) for _ in range(1, n_layers)
        ])
        self.pool = global_mean_pool
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # Global pooling to get graph-level embeddings
        x = self.pool(x, batch)
        x = F.sigmoid(self.fc1(x))
        embeddings = self.fc2(x)
        return embeddings

# Decoder: Predicts graph-level attributes using embeddings
class Decoder(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, embeddings):
        x = F.relu(self.fc1(embeddings))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
