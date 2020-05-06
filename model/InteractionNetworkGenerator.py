import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class InteractionNetworkGenerator(nn.Module):
    def __init__(self, effect_dims=4, hidden_size=128, latent_size=2):
        super().__init__()
        # setup the two linear transformations used
        self.node_dims = 7+latent_size #X+z_node
        self.z_edge_dims = latent_size
        self.z_u_dims = latent_size
        self.effect_dims = effect_dims
        self.hidden_size = hidden_size
        self.relational_model = RelationalModel(2*self.node_dims + self.z_edge_dims, 
                                                self.effect_dims, self.hidden_size)
        self.object_model     = DynamicsModel(self.node_dims+self.effect_dims+self.z_u_dims,
                                              self.hidden_size)
    def forward(self, x, z, edge_index):
        # x shape: [bs, steps, nodes, node_dims]
        bs, steps, nodes, node_dims = x.size()
        sender_relations = F.one_hot(edge_index[0]).T.unsqueeze(0).repeat(bs*steps,1,1).type(torch.float)
        receiver_relations = F.one_hot(edge_index[1]).T.unsqueeze(0).repeat(bs*steps,1,1).type(torch.float)
        _, edges = edge_index.size()
        row, col = edge_index
        # define the forward computation on the latent z
        z_node, z_edge, z_global = z
        z_node = torch.unsqueeze(z_node, dim=1).repeat(1, steps, 1, 1)
        z_edge = z_edge.repeat(steps, 1, 1)
        z_global = torch.unsqueeze(z_global, dim=1).repeat(steps, nodes, 1)
        x = torch.cat((x, z_node), dim=-1).view(bs*steps, nodes, -1)
        senders   = sender_relations.permute(0, 2, 1).bmm(x)
        receivers = receiver_relations.permute(0, 2, 1).bmm(x)
        effects = self.relational_model(torch.cat([senders, receivers, z_edge], -1))
        effect_receivers = receiver_relations.bmm(effects)
        # predicted shape [bs*steps, nodes, 2]
        predicted = self.object_model(torch.cat([x, z_global,effect_receivers], -1))
        return predicted.view(bs, steps, nodes, 2)

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x
    
class DynamicsModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2), #speedX and speedY
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)