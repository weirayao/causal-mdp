import torch
import torch.nn as nn
import torch.nn.functional as F
from InteractionNetworkGenerator import InteractionNetworkGenerator
from GNNLSTMRecgnition import GNNLSTMRecgnition

class PhysicsVAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=32, effect_dims=32, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = GNNLSTMRecgnition(hidden_size=32, latent_size=z_dim)
        self.decoder = InteractionNetworkGenerator(effect_dims=effect_dims, hidden_size=128, latent_size=z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
    
    def encode(self, xy, edge_index):
        return self.encoder(xy, edge_index)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, x, z, edge_index):
        return self.decoder(x, z, edge_index)

    def forward(self, x, y, edge_index):
        xy = torch.cat((x, y), dim=-1)
        self.z_stats = self.encode(xy, edge_index)
        z_node_mu, z_node_logvar, z_edge_mu, z_edge_logvar, z_global_mu, z_global_logvar = self.z_stats
        z_node = self.reparameterize(z_node_mu, z_node_logvar)
        z_edge = self.reparameterize(z_edge_mu, z_edge_logvar)
        z_global = self.reparameterize(z_global_mu, z_global_logvar)
        z_sample = [z_node, z_edge, z_global]
        return self.decode(x, z_sample, edge_index), self.z_stats
    
    def inference(self, x_n, y_n, x_test, edge_index):
        # x_n, y_n [n, steps, n_objects, node_dims ]
        # x_test shape [1, steps, n_objects, node_dims]
        with torch.no_grad():
            # adapting posterior with x_n
            _, _ = self.forward(x_n, y_n, edge_index)
            z_node_mu, z_node_logvar, z_edge_mu, z_edge_logvar, z_global_mu, z_global_logvar = self.z_stats
            # generation process
            ez_node_logvar = torch.mean(z_node_logvar, dim=0)
            ez_edge_mu = torch.mean(z_edge_mu, dim=0)
            ez_edge_logvar = torch.mean(z_edge_logvar, dim=0)
            ez_global_mu = torch.mean(z_global_mu, dim=0)
            ez_global_logvar = torch.mean(z_global_logvar, dim=0)
            z_node = self.reparameterize(ez_node_mu, ez_node_logvar).unsqueeze(0)
            z_edge = self.reparameterize(ez_edge_mu, ez_edge_logvar).unsqueeze(0)
            z_global = self.reparameterize(ez_global_mu, ez_global_logvar).unsqueeze(0)
            z_sample = [z_node, z_edge, z_global]
            pred = self.decode(x_test, z_sample, edge_index)
        return pred