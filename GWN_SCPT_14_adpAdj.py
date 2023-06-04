'''
GraphWaveNet.py
'''
import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import numpy as np
import pandas as pd
from Utils import load_pickle

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 dropout=0.0,
                 in_dim=2,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 sga = True,
                 adp_adj = False):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.adp_adj = adp_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1

        # SGA: spatially gated addition
        if sga:
            self.sga_gate1 = nn.Conv2d(in_channels=2*residual_channels, out_channels=4*residual_channels, kernel_size=(1,1))
            self.sga_gate2 = nn.Conv2d(in_channels=4*residual_channels, out_channels=1, kernel_size=(1,1))
            self.addition = self.spatially_gated_addition
        else:
            self.addition = self.naive_addition

        # linear projections from embedding to adaptive adjacency
        if adp_adj:
            self.embed_proj11 = nn.Linear(residual_channels,4*residual_channels, bias=False)
            self.embed_proj12 = nn.Linear(residual_channels*4,residual_channels, bias=False)
            self.embed_proj21 = nn.Linear(residual_channels,4*residual_channels, bias=False)
            self.embed_proj22 = nn.Linear(residual_channels*4,residual_channels, bias=False)
    
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=2 + int(adp_adj)))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def spatially_gated_addition(self, x, e):
        # x [BDNL] is latent representation
        # e [DN] is embedding
        # only one set, because this is just a quality gate for the embedding
        # they all should have the same quality.
        e = e.unsqueeze(0).unsqueeze(-1).expand(x.shape[0],-1,-1,x.shape[-1]) # BDNL
        g = self.sga_gate1(torch.cat((x,e), dim=1))
        g = F.relu(g)
        g = self.sga_gate2(g)
        g = torch.sigmoid(g)
        x = x + g * e
        return x

    def naive_addition(self, x, e):
        return x + e.unsqueeze(0).unsqueeze(-1).expand(x.shape[0],-1,-1,x.shape[-1]) # BDNL

    def forward(self, input, adj, embed):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # adaptive adjacency based on SCPT embeddings
        if self.adp_adj:
            # without SCPT, embed=0 anyway, since the projs have no bias, then it is all zeroes anyway.
            # nv = nodevec
            nv1 = self.embed_proj12(F.relu(self.embed_proj11(embed.T))) # embed = [D,N]
            nv2 = self.embed_proj22(F.relu(self.embed_proj21(embed.T))) # embed = [D,N]
            adp = F.softmax(F.relu(torch.mm(nv1, nv2.T)), dim=1) # adp = [N,N]
            adj = adj + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            x = self.addition(x, embed)
            # dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # GCN
            x = self.addition(x, embed)
            x = self.gconv[i](x, adj)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return np.array(d_mat.dot(adj).astype(np.float32).todense())

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_adj(pkl_filename, adjtype, dataname):
    if dataname == 'METRLA' or dataname == 'PEMSBAY' or dataname=='PEMS11160':
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    elif dataname == 'PEMSD7M':
        adj_mx = pd.read_csv(pkl_filename).values
        distances = adj_mx[~np.isinf(adj_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(adj_mx / std))
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def nt_xent_loss(out_1, out_2, temperature):
    """Loss used in SimCLR."""
    # https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/losses/self_supervised_learning.py
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss

class Contrastive_FeatureExtractor_conv(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.conv1 = torch.nn.Conv1d( 1, 32, 13, stride=1) # 1 hour --> per timestep
        self.conv2 = torch.nn.Conv1d(32, 32, 12, stride=12) # 2 hour --> per hour
        self.conv3 = torch.nn.Conv1d(32, 32, 24, stride=24) # 1 day --> per day
        self.fc1 = torch.nn.Linear(32*3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32*3)
        self.bn4 = torch.nn.BatchNorm1d(32)
    def forward(self, x):
        x = self.conv1(x[:,None,:])
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        # sample half of samples
        n_half = int(x.shape[-1]/2)
        x_ = torch.empty(x.shape[0], x.shape[1], n_half).to(x.device)
        for i in range(x.shape[0]):
            idx = np.arange(x.shape[2])
            np.random.shuffle(idx)
            idx = idx < n_half
            x_[i, :, :] = x[i, :, idx]
        # aggregate
        x_u = x_.mean(axis=2)
        x_z = x_.std(axis=2)
        x_x, _ = torch.max(x_, axis=2)
        x = torch.cat((x_u, x_z, x_x), axis=1)
        # project
        x = self.bn3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn4(x)
        return x
    
    def contrast(self, x):
        # project
        x1 = self(x)
        x2 = self(x)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        # L2 norm
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        # calculate loss
        return nt_xent_loss(x1,x2,self.temperature)

def main(GPU=None):
    from Param import CHANNEL,N_NODE,TIMESTEP_IN
    from Param_GraphWaveNet import ADJPATH,ADJTYPE
    if GPU == None:
        GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu") 
    model = gwnet(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=None).to(device)
    return model
    
if __name__ == '__main__':
    main()
