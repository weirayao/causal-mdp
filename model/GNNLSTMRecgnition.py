import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean

class GNNLSTMRecgnition(nn.Module):
    def __init__(self, hidden_size=16, latent_size=2):
        super().__init__()
        ## GNN ##
        self.node_dims = 7+2 #X+y
        self.hidden_size=hidden_size
        self.edge_dims=hidden_size
        self.u_dims=hidden_size 
        edge_model = EdgeModel(self.node_dims, self.edge_dims, self.u_dims, self.hidden_size)
        node_model = NodeModel(self.node_dims, self.edge_dims, self.u_dims, self.hidden_size)
        global_model = GlobalModel(self.node_dims, self.u_dims, self.hidden_size)
        self.op = MetaLayer(edge_model, node_model, global_model)
        ## LSTM ##
        self.edge_rnn = nn.LSTM(self.edge_dims, self.hidden_size, batch_first=True)
        self.node_rnn = nn.LSTM(self.node_dims, self.hidden_size, batch_first=True)
        self.global_rnn = nn.LSTM(self.u_dims, self.hidden_size, batch_first=True)
        ## Linear ##
        self.edge_fc_1 = nn.Linear(self.hidden_size, latent_size)
        self.edge_fc_2 = nn.Linear(self.hidden_size, latent_size)
        self.node_fc_1 = nn.Linear(self.hidden_size, latent_size)
        self.node_fc_2 = nn.Linear(self.hidden_size, latent_size)
        self.global_fc_1 = nn.Linear(self.hidden_size, latent_size)
        self.global_fc_2 = nn.Linear(self.hidden_size, latent_size)

    def forward(self, x, edge_index):
        # define the forward computation on the latent z
        # x shape: [n_experiments, steps, nodes, node_dims]
        bs, steps, nodes, node_dims = x.size()
        _, edges = edge_index.size()
        x_reshape = x.view(-1, nodes, node_dims)
        node_attr, edge_attr, global_attr = self.op(x_reshape, edge_index)
        node_attr = node_attr.view(bs, steps, nodes, self.node_dims).permute(0,2,1,3)
        edge_attr = edge_attr.view(bs, steps, edges, self.edge_dims).permute(0,2,1,3)
        global_attr = global_attr.view(bs, steps, self.u_dims)
        node_attr = node_attr.reshape(-1, steps, self.node_dims)
        edge_attr = edge_attr.reshape(-1, steps, self.edge_dims)
        # RNN forward
        node_out, _ = self.node_rnn(node_attr)
        edge_out, _ = self.edge_rnn(edge_attr)
        global_out, _ = self.global_rnn(global_attr)
        # get the outputs of the last time step
        node_out  = node_out[:, -1, :]
        edge_out = edge_out[:, -1, :]
        global_out = global_out[:, -1, :]
        # FC forward
        z_node_mu = self.node_fc_1(node_out).view(bs, nodes, -1)
        z_node_logvar = self.node_fc_2(node_out).view(bs, nodes, -1)
        z_edge_mu = self.edge_fc_1(edge_out).view(bs, edges, -1)
        z_edge_logvar = self.edge_fc_2(edge_out).view(bs, edges, -1)
        z_global_mu = self.global_fc_1(global_out).view(bs, -1)
        z_global_logvar = self.global_fc_2(global_out).view(bs, -1)
        return [z_node_mu, z_node_logvar, z_edge_mu, z_edge_logvar, z_global_mu, z_global_logvar]

class MetaLayer(torch.nn.Module):
    """A meta layer for building any kind of graph network, inspired by the
    Relational Inductive Biases, Deep Learning, and Graph Networks
    <https://arxiv.org/abs/1806.01261>`_ paper.

    Args:
        edge_model (Module, optional): A callable which updates a graph's edge
            features based on its source and target node features, its current
            edge features and its global features. (default: :obj:`None`)
        node_model (Module, optional): A callable which updates a graph's node
            features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (Module, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features.
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index):
        """"""
        row, col = edge_index
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[:,row,:], x[:,col,:])
        
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)
        if self.global_model is not None:
            u = self.global_model(x)
        return x, edge_attr, u

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)
                            
class EdgeModel(nn.Module):
    def __init__(self, node_dims, edge_dims, u_dims, hidden_size=32):
        super().__init__()
        input_size = 2*node_dims
        self.edge_mlp = Seq(Lin(input_size, hidden_size), ReLU(), Lin(hidden_size, edge_dims))

    def forward(self, src, dest):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest], dim=-1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, node_dims, edge_dims, u_dims, hidden_size=32):
        super().__init__()
        mlp_1_input_size = node_dims+edge_dims
        self.node_mlp_1 = Seq(Lin(mlp_1_input_size, hidden_size), ReLU(), Lin(hidden_size, hidden_size))
        mlp_2_input_size = node_dims+hidden_size
        self.node_mlp_2 = Seq(Lin(mlp_2_input_size, hidden_size), ReLU(), Lin(hidden_size, node_dims))

    def forward(self, x, edge_index, edge_attr):
        # x: [bs, N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [bs, E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[:,row,:], edge_attr], dim=-1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=1, dim_size=x.size(1))
        out = torch.cat([x, out], dim=-1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, node_dims, u_dims, hidden_size=32):
        super().__init__()
        input_size = node_dims
        self.global_mlp = Seq(Lin(input_size, hidden_size), ReLU(), Lin(hidden_size, u_dims))

    def forward(self, x):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.mean(x, dim=1, keepdim=False)
        #out = scatter_mean(x, batch, dim=0)
        return self.global_mlp(out)