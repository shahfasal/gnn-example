from datasetPyGRDF import DatasetPyGRDF
import torch

from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear
import argparse
import torch_geometric.transforms as T

from torch_geometric.transforms import ToUndirected, RandomLinkSplit

from torch_geometric.nn import SAGEConv, to_hetero
parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = DatasetPyGRDF('data/')
data = dataset[0].to(device)
# data['if-gnn:Class1'].x = torch.eye(data['if-gnn:Class1'].num_nodes, device=device)
# del data['if-gnn:Class1'].num_node

data = ToUndirected()(data)
print("ToUndirected",data)
# del data['if-gnn:Class2', 'rev_:isConnectedTo', 'if-gnn:Class1'].edge_label

train_data, val_data, test_data =  T.RandomLinkSplit(
num_val=0.05,
num_test=0.1,
neg_sampling_ratio=0.5,
edge_types=[('if-gnn:Class1', 'if-gnn:isConnectedTo', 'if-gnn:Class2')],
rev_edge_types=[('if-gnn:Class2', 'rev_if-gnn:isConnectedTo', 'if-gnn:Class1')]
)(data)


if args.use_weighted_loss:
    weight = torch.bincount(train_data['if-gnn:Class1','if-gnn:Class2'].edge_label)
    weight = weight.max() / weight
else:
    weight = None

def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
#
#
#
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index

        z = torch.cat([z_dict['if-gnn:Class1'][row], z_dict['if-gnn:Class2'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)

# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['if-gnn:Class1','if-gnn:Class2'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['if-gnn:Class1','if-gnn:Class2'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)

def train():
    model.train()
    optimizer.zero_grad()

    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['if-gnn:Class1','if-gnn:Class2'].edge_label_index)
    target = train_data['if-gnn:Class1','if-gnn:Class2'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

for epoch in range(1, 30):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
