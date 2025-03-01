import torch
from graph_library.gin_torch_models import GINEncoder
from graph_library.gsage_torch_models import GraphSAGE


def InitializeEncoderModel(model_type, input_dim, hidden_dim, n_layers, embedding_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize encoder
    statement = model_type+"(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers,  embedding_dim=embedding_dim).to(device)"
    encoder = eval(statement)

    return encoder


def GetEmbeddings(encoder, modelPath, data):
    encoder.load_state_dict(torch.load(modelPath))
    encoder.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate embeddings for a batch of graphs
    data = data.to(device)

    #embeddings = encoder(data.x, data.edge_index, data.batch)
    embeddings = encoder(data)

    return embeddings
