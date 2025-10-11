import torch
import torch.nn as nn
import math
import pdb
import copy
from numpy import inf
# from sparsemax import Sparsemax
import torch.nn.functional as F
import math
from attpool import AttPool, PaiAttPool, PaiAttPool2, PaiAttPool3
from math import ceil, sqrt
from device import device
import numpy as np


class FeaStConv(nn.Module): ## FeaStConv
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='relu',bias=True): # ,device=None):
        super(FeaStConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.heads = num_neighbor
        self.bias = nn.Parameter(torch.Tensor(out_c))
        self.mlp = nn.Linear(in_c, self.heads) 
        self.mlp_out = nn.Linear(in_c, self.heads * out_c, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0

        self.reset_parameters()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()
    
    @staticmethod
    def normal(tensor, mean, std):
        if tensor is not None:
            tensor.data.normal_(mean, std)

    def reset_parameters(self):
        self.normal(self.bias, mean=0, std=0.1)

    def forward(self,x,t_vertex,neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize*num_pts, num_neighbor, feats)
        #### relative position ####
        x_relative = x_neighbors - x_neighbors[:, 0:1, :]

        q = self.softmax(self.mlp(x_relative.view(-1, feats))).view(bsize, num_pts, num_neighbor*self.heads, -1)
        x_j = self.mlp_out(x_neighbors.view(-1, feats)).view(bsize, num_pts, num_neighbor*self.heads, -1)
        out_feat =  (x_j * q).sum(dim=2) + self.bias
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return self.activation(out_feat)


class AdaptiveConv(nn.Module):
    """AdaptiveConv block from AdaptConv (Zhang et al., CVPR 2021)."""

    def __init__(self, in_channels, out_channels, feat_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels * in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        """Apply adaptive convolution.

        Args:
            x: Tensor of shape (batch, in_channels, num_points, k).
            y: Tensor of shape (batch, feat_channels, num_points, k).
        """

        batch_size, _, num_points, k = x.size()

        y = self.conv0(y)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels, self.in_channels)

        x = x.permute(0, 2, 3, 1).unsqueeze(4)
        x = torch.matmul(y, x).squeeze(4)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x


class AdaptiveConvBlock(nn.Module):
    """Wrapper that adapts AdaptiveConv to the mesh autoencoder interface."""

    def __init__(self, num_pts, in_c, num_neighbor, out_c, activation='elu'):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_neighbor = num_neighbor

        self.adaptive_conv = AdaptiveConv(in_channels=6, out_channels=out_c, feat_channels=2 * in_c)
        self.mlp_out = nn.Linear(in_c, out_c)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0, -1, 0] = 0.0

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, t_vertex, neighbor_index):
        bsize, num_pts, feats = x.size()
        num_neighbor = min(self.num_neighbor, neighbor_index.size(2))

        x = x * self.zero_padding.to(x.device)

        neighbor_index = neighbor_index[:, :, :num_neighbor].contiguous()
        flat_neighbors = neighbor_index.view(bsize * num_pts * num_neighbor).long()
        batch_index = torch.arange(bsize, device=x.device).view(-1, 1).repeat(1, num_pts * num_neighbor).view(-1)

        x_neighbors = x[batch_index, flat_neighbors, :].view(bsize, num_pts, num_neighbor, feats)
        center_feat = x.unsqueeze(2).expand(-1, -1, num_neighbor, -1)
        feat_input = torch.cat((x_neighbors - center_feat, center_feat), dim=-1)
        feat_input = feat_input.permute(0, 3, 1, 2).contiguous()

        pos = t_vertex
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.size(0) == 1 and pos.size(0) != bsize:
            pos = pos.expand(bsize, -1, -1)
        if pos.size(0) != bsize:
            raise ValueError(f"AdaptiveConvBlock expected position batch {bsize}, got {pos.size(0)}")
        pos = pos.to(x.device)
        pos_dim = pos.size(-1)
        if pos_dim < 3:
            pad = torch.zeros(bsize, pos.size(1), 3 - pos_dim, device=pos.device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=-1)
        pos = pos[..., :3].clone()
        if pos.size(1) < num_pts:
            pad = torch.zeros(bsize, num_pts - pos.size(1), 3, device=pos.device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=1)
        elif pos.size(1) > num_pts:
            pos = pos[:, :num_pts, :]
        pos[:, -1, :] = 0

        pos_neighbors = pos[batch_index, flat_neighbors, :].view(bsize, num_pts, num_neighbor, 3)
        center_pos = pos.unsqueeze(2).expand(-1, -1, num_neighbor, -1)
        pos_input = torch.cat((pos_neighbors - center_pos, center_pos), dim=-1)
        pos_input = pos_input.permute(0, 3, 1, 2).contiguous()

        out = self.adaptive_conv(pos_input, feat_input)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.permute(0, 2, 1).contiguous()

        res = self.mlp_out(x.view(-1, self.in_c)).view(bsize, num_pts, self.out_c)
        out = out + res
        out = out * self.zero_padding.to(out.device)
        return self.activation(out)


class PaiConv(nn.Module): ## LSA-Conv
    '''
    Gao, Zhongpai, et al. "Learning local neighboring structure for robust 3D shape representation." 
    Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 2. 2021.
    '''
    def __init__(self, num_pts, in_c, num_neighbor, out_c, activation='elu',bias=True): # ,device=None):
        super(PaiConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        self.adjweight = nn.Parameter(torch.randn(num_pts, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0
        self.mlp_out = nn.Linear(in_c, out_c)
        # self.sparsemax = Sparsemax(dim=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, t_vertex, neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)
        # x_neighbors = x_neighbors.view(num_pts, bsize*feats, num_neighbor)     
        # weight = self.softmax(torch.bmm(torch.transpose(x_neighbors, 1, 2), x_neighbors))
        # x_neighbors = torch.bmm(x_neighbors, weight) #.view(num_pts, feats, num_neighbor)
        x_neighbors = torch.einsum('bnkf, bnkt->bntf', x_neighbors, self.adjweight[None].repeat(bsize, 1, 1, 1))   #self.sparsemax(self.adjweight))
        x_neighbors = self.activation(x_neighbors.contiguous().view(bsize*num_pts, num_neighbor*feats)) 
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        x_res = self.mlp_out(x.view(-1, self.in_c)).view(bsize, -1, self.out_c)
        return out_feat + x_res

class PaiConvTiny(nn.Module): ## SDConv or HSDConv
    '''
    Gao, Z., Yan, J., Zhai, G. and Yang, X., 2021, August. 
    Learning Spectral Dictionary for Local Representation of Mesh. In IJCAI (pp. 685-692).
    '''
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='relu',bias=True,base_size=32): # ,device=None):
        super(PaiConvTiny,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        # self.norm = nn.BatchNorm1d(in_c)
        # self.fc1 = nn.Linear(in_c, in_c)
        # self.fc2 = nn.Linear(out_c, out_c)
        mappingsize = 64
        self.num_base = base_size
        self.num_neighbor = num_neighbor
        if num_pts > 128:
            num_base = self.num_base
            self.temp_factor = 100
            self.tmptmlp = nn.Linear(mappingsize*2, 1)
            self.softmax = nn.Softmax(dim=1) # Sparsemax(dim=-1) # nn.Softmax(dim=1)
            self.mlp = nn.Linear(mappingsize*2, num_base)
        else:
            num_base = num_pts

        self.mlp_out = nn.Linear(in_c, out_c)
        self.adjweight = nn.Parameter(torch.randn(num_base, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, t_vertex, neighbor_index):
        bsize, num_pts, feats = x.size()
        neighbor_index = neighbor_index[:, :, :self.num_neighbor].contiguous()
        _, _, num_neighbor = neighbor_index.size()

        # x = self.activation(self.fc1(x.view(-1, feats))).view(bsize, num_pts, -1)
        # x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)

        if num_pts > 128:
            tmpt = torch.sigmoid(self.tmptmlp(t_vertex))*(0.1 - 1.0/self.temp_factor) + 1.0/self.temp_factor 
            adjweightBase = self.softmax(self.mlp(t_vertex)/tmpt)
            adjweight = torch.einsum('ns,skt->nkt', adjweightBase, self.adjweight)[None].repeat(bsize, 1, 1, 1)
        else:
            adjweight = self.adjweight[None].repeat(bsize, 1, 1, 1)
        x_neighbors = torch.einsum('bnkf,bnkt->bnft', x_neighbors, adjweight)
        x_neighbors = F.elu(x_neighbors.view(bsize*num_pts, num_neighbor*feats))
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        # out_feat = self.activation(self.fc2(out_feat.view(-1, self.out_c))).view(bsize, num_pts, -1)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        x_res = self.mlp_out(x.view(-1, self.in_c)).view(bsize, -1, self.out_c)
        return out_feat + x_res
    

class PaiAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, 
                 t_vertices, sizes, num_neighbors, x_neighbors, D, U, activation = 'elu', 
                 is_hierarchical=True, is_old_filter=False, base_size=32, use_feast_ablation=False,
                 use_adaptive_ablation=False):
        super(PaiAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.x_neighbors = [torch.cat([torch.cat([torch.arange(x.shape[0]-1), torch.tensor([-1])]).unsqueeze(1), x], 1) for x in x_neighbors]
        #self.x_neighbors = [x.float().cuda() for x in x_neighbors]
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.num_neighbors = num_neighbors
        self.D = [nn.Parameter(x, False) for x in D]
        self.D = nn.ParameterList(self.D)
        self.U = [nn.Parameter(x, False) for x in U]
        self.U = nn.ParameterList(self.U)
        self.is_hierarchical = is_hierarchical
        self.is_old_filter = is_old_filter
        self.use_feast_ablation = use_feast_ablation
        self.base_size = base_size
        self.use_adaptive_ablation = use_adaptive_ablation
        if self.use_adaptive_ablation and self.use_feast_ablation:
            raise ValueError('AdaptiveConv and FeaSt ablations cannot both be enabled')

        mappingsize = 64
        self.o_vertices = t_vertices
        if not self.is_old_filter:
        ## v1
            self.B_1 = nn.Parameter(torch.randn(24, mappingsize//2) , requires_grad=False)
            self.B_2 = nn.Parameter(torch.randn(3, mappingsize//2) , requires_grad=False)  
            self.t_vertices = [torch.cat([(x[self.x_neighbors[i]][:, 1:] - x[:, None]).view(x.shape[0], -1), x], dim=1) for i, x in enumerate(t_vertices)]
            self.t_vertices = [((x - x.min(dim=0, keepdim=True)[0]) / (x.max(dim=0, keepdim=True)[0] \
                                - x.min(dim=0, keepdim=True)[0]) - 0.5)*5 for x in self.t_vertices]
            self.t_vertices = [torch.cat([2.*math.pi*x[:, :24] @ (self.B_1.data).to(x),
                                        2.*math.pi*x[:, 24:] @ (self.B_2.data).to(x)], dim=1) for x in self.t_vertices]
        else:
        ## v2
            self.B = nn.Parameter(torch.randn(6, mappingsize) , requires_grad=False)
            self.t_vertices = [torch.cat([x[self.x_neighbors[i]][:, 1:].mean(dim=1) - x, x], dim=-1) for i, x in enumerate(t_vertices)]
            self.t_vertices = [2.*math.pi*x @ (self.B.data).to(x) for x in self.t_vertices]
            self.t_vertices = [((x - x.min(dim=0, keepdim=True)[0]) / (x.max(dim=0, keepdim=True)[0] \
                                - x.min(dim=0, keepdim=True)[0]) - 0.5)*100 for x in self.t_vertices]
        
        self.t_vertices = [torch.cat([torch.sin(2*x), torch.cos(x)], dim=-1) for x in self.t_vertices]
        
        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = activation
        self.conv = []
        input_size = filters_enc[0]
        def make_conv_layer(num_pts, in_c, num_neighbor, out_c, act):
            if self.use_adaptive_ablation:
                return AdaptiveConvBlock(num_pts, in_c, num_neighbor, out_c, activation=act)
            if (not self.is_hierarchical) and self.use_feast_ablation:
                return FeaStConv(num_pts, in_c, num_neighbor, out_c, activation=act)
            return PaiConvTiny(num_pts, in_c, num_neighbor, out_c, activation=act, base_size=self.base_size)
        for i in range(len(num_neighbors)-1):
            self.conv.append(make_conv_layer(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[i+1],
                                             self.activation))
            input_size = filters_enc[i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0])
        
        self.dconv = []
        input_size = filters_dec[0]
        for i in range(len(num_neighbors)-1):
            self.dconv.append(make_conv_layer(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[i+1],
                                              self.activation))
            input_size = filters_dec[i+1]  

            if i == len(num_neighbors)-2:
                input_size = filters_dec[-2]
                self.dconv.append(make_conv_layer(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[-1],
                                                   'identity'))
                    
        self.dconv = nn.ModuleList(self.dconv)

        ref_mean = torch.mean(t_vertices[0], dim=0)
        ref_std = torch.mean(t_vertices[0], dim=0)
        nV_ref = [0.1 * (t_vertices[i] - ref_mean)/ref_std for i in range(len(t_vertices))]

        self.attpoolenc, self.attpooldec = self.init_attpool(nV_ref)
        self.attpoolenc = torch.nn.ModuleList(self.attpoolenc)
        self.attpooldec = torch.nn.ModuleList(self.attpooldec)

    def init_attpool(self, V_ref=None):
        attpooldec = []
        attpoolenc = []
        config = {'prior_init': True, 'prior_coef': 0.8, 'dim_att': 32, 'top_k_enc': 8, 'top_k_dec': 32}
        if V_ref is not None:
            for i in range(len(V_ref)-1):
                idx_k, idx_q = i, i+1
                k_init, q_init = V_ref[idx_k].t(), V_ref[idx_q].t()
                config_enc = config
                config_enc['num_k'], config_enc['num_q'] = k_init.size(1), q_init.size(1)
                config_enc['top_k'] = config['top_k_enc'][min(i,len(config['top_k_enc'])-1)] \
                    if isinstance(config['top_k_enc'],list) else config['top_k_enc']
                attpoolenc.append(PaiAttPool2(config=config_enc, const_k=k_init, const_q=q_init,
                                          prior_map=self.D[i].to_dense()))
                config_dec = config
                config_dec['num_k'], config_dec['num_q'] = q_init.size(1), k_init.size(1)
                config_dec['top_k'] = config['top_k_dec'][min(i,len(config['top_k_dec'])-1)] \
                    if isinstance(config['top_k_dec'],list) else config['top_k_dec']
                attpooldec.append(PaiAttPool2(config=config_dec, const_k=q_init, const_q=k_init,
                                          prior_map=self.U[i].to_dense()))
        return attpoolenc, attpooldec

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        x = x.permute(1, 2, 0).contiguous()  #M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N

        x = torch.spmm(L, x)  # Mp x Fin*N
        x = x.view(Mp, Fin, N)  # Mp x Fin x N
        x = x.permute(2, 0, 1).contiguous()   # N x Mp x Fin
        return x

    def encode(self,x):
        bsize = x.size(0)
        S = self.x_neighbors
        D = self.D
        t_vertices = self.t_vertices
        for i in range(len(self.num_neighbors)-1):
            t_arg = self.o_vertices[i] if self.use_adaptive_ablation else t_vertices[i]
            x = self.conv[i](x, t_arg, S[i].repeat(bsize,1,1))
            # x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i]) if not self.is_hierarchical else self.attpoolenc[i](x) #, t_vertices[i], t_vertices[i+1]) #, t_vertices[i+1])
            # self.t_vertices[i+1] = self.attpoolenc[i](self.t_vertices[i][None]).squeeze().detach() #, t_vertices[i])#, t_vertices[i+1])
        # x = self.conv[-1](x, t_vertices[-1], S[-1].repeat(bsize,1,1))
        # x[5] = x[2]
        # x[5,4:6] = x[21, 4:6]  ## 34
        # x[5,12:15] = x[21, 12:15]
        # x[5,17:20] = x[34, 17:20]  ## 34
        # x[5,23:27] = x[34, 23:27]
        x = x.view(bsize,-1)
        # x = x[:, :-1].view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.x_neighbors
        U = self.U
        t_vertices = self.t_vertices

        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)

        for i in range(len(self.num_neighbors)-1):
            # x = torch.matmul(U[-1-i],x)
            x = self.poolwT(x, U[-1-i]) if not self.is_hierarchical else self.attpooldec[-i-1](x) #, t_vertices[-i-1], t_vertices[-i-2]) #, t_vertices[-i-2])
            t_arg = self.o_vertices[-2-i] if self.use_adaptive_ablation else t_vertices[-2-i]
            x = self.dconv[i](x, t_arg, S[-2-i].repeat(bsize,1,1))
        final_t = self.o_vertices[0] if self.use_adaptive_ablation else t_vertices[0]
        x = self.dconv[-1](x, final_t, S[0].repeat(bsize,1,1))
        return x

    def update(self):

        for i in range(len(self.num_neighbors)-1):
            self.o_vertices[i+1] = self.attpoolenc[i](self.o_vertices[i][None]).squeeze() #, self.t_vertices[i].detach(), self.t_vertices[i+1].detach()).squeeze() # , self.t_vertices[i].detach()).squeeze()
        
        if not self.is_old_filter:
        ## v1
            self.t_vertices = [torch.cat([(x[self.x_neighbors[i]][:, 1:] - x[:, None]).view(x.shape[0], -1), x], dim=1) for i, x in enumerate(self.o_vertices)]
            self.t_vertices = [((x - x.min(dim=0, keepdim=True)[0]) / (x.max(dim=0, keepdim=True)[0] \
                                 - x.min(dim=0, keepdim=True)[0]) - 0.5)*5 for x in self.t_vertices]
            self.t_vertices = [torch.cat([2.*math.pi*x[:, :24] @ (self.B_1.data).to(x),
                                          2.*math.pi*x[:, 24:] @ (self.B_2.data).to(x)], dim=1) for x in self.t_vertices]
        else:
        ## v2
            self.t_vertices = [torch.cat([x[self.x_neighbors[i]][:, 1:].mean(dim=1) - x, x], dim=-1) for i, x in enumerate(self.o_vertices)]
            self.t_vertices = [2.*math.pi*x @ (self.B.data).to(x) for x in self.t_vertices]
            self.t_vertices = [((x - x.min(dim=0, keepdim=True)[0]) / (x.max(dim=0, keepdim=True)[0] \
                                - x.min(dim=0, keepdim=True)[0]) - 0.5)*100 for x in self.t_vertices]
        self.t_vertices = [torch.cat([torch.sin(x), torch.cos(x)], dim=-1) for x in self.t_vertices]
        return

    def forward(self,x):
        bsize = x.size(0)
        # alpha = torch.linspace(0, 1, steps=10).cuda().unsqueeze(1)
        if self.is_hierarchical:
            self.update()
        z = self.encode(x)
        # z[0] = z[18] - z[7] + z[8]
        # z[0], z[1] = z[18], z[29] ## z[23], z[26]
        # z[2:7] = alpha * z[0:1] + (1-alpha)*z[1:2]
        x = self.decode(z)
        return x
