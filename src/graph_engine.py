import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class AxonGraphGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(AxonGraphGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def detect_lateral_movement(node_features, edge_list):
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    model = AxonGraphGNN(num_features=x.shape[1])
    model.eval()
    out = model(data)
    return out.argmax(dim=1)