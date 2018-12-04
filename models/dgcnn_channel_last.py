import os, sys
import time
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import nn_utils

def get_edge_features(x, k):
    """
    Args:
        x: point cloud [B, N, dims]
        k: kNN neighbours
    Return:
        [B, N, k, 2*dims]    
    """

    B, N, dims = x.shape
    # batched pair-wise distance
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(x, xt)
    xs = torch.sum(x**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id    
    _, idx = torch.sort(dist, dim=2)
    idx = idx[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous()
    idx = idx.view(B, N*k)

    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 0, idx[b]) # [N*k, d] <- [N, d], 0, [N*k]
        tmp = tmp.view(N, k, dims)
        neighbors.append(tmp)
    neighbors = torch.stack(neighbors) # [B, N, k, d]

    # centralize
    central = x.unsqueeze(2) # [B, N, 1, d]
    central = central.repeat(1, 1, k, 1) # [B, N, k, d]

    ee = torch.cat([central, neighbors-central], dim=3)
    return ee

class edgeConv(nn.Module):
    """ Edge Convolution using fully-connected h
    [B, N, Fin] -> [B, N, Fout]
    """
    def __init__(self, Fin, Fout, k):
        super(edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        #self.fc = nn_utils.fcbr(2*Fin, Fout)
        self.fc = nn.Linear(2*Fin, Fout)

    def forward(self, x):
        B, N, Fin = x.shape
        
        x = get_edge_features(x, self.k); # [B, N, k, 2Fin]
        
        #x = x.view(-1, 2*Fin)
        x = self.fc(x) # [B, N, k, Fout]
        #x = x.view(B, N, self.k, -1)
        
        x, _ = torch.max(x, 2) # [B, N, Fout]

        return x

class edgeConvC(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, N, Fin] -> [B, N, Fout]
    """
    def __init__(self, Fin, Fout, k):
        super(edgeConvC, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv = nn.Conv2d(2*Fin, Fout, 1)
        #self.fc = nn.Linear(2*Fin, Fout)

    def forward(self, x):
        B, N, Fin = x.shape
        
        x = get_edge_features(x, self.k); # [B, N, k, 2Fin]
        
        x = x.permute(0, 3, 1, 2) # [B, 2Fin, N, k]
        x = self.conv(x) # [B, Fout, N, k]

        x, _ = torch.max(x, 3) # [B, Fout, N]
        x = x.permute(0, 2, 1)  # [B, N, Fout]

        assert x.shape == (B, N, self.Fout)

        return x


class dgcnn(nn.Module):
    """ Classification architecture
    [B, N, F] -> [B, nCls]
    """
    def __init__(self, conf):
        super(dgcnn, self).__init__()
        self.ec0 = edgeConvC(conf.Fin, 64, conf.k)
        self.ec1 = edgeConvC(64, 128, conf.k)
        self.conv0 = nn.Conv1d(128+64, 512, 1, 1)
        self.fc0 = nn.Linear(512, conf.nCls)


    def forward(self, x):
        B, N, Fin = x.shape
        x = self.ec0(x) # [B, N, 64]
        x1 = x
        x = self.ec1(x) # [B, N, 128]
        x2 = x

        x = torch.cat((x1, x2), dim=2) # [B, N, 64+128]

        x = x.permute(0, 2, 1) # [B, 64+128, N]
        x = self.conv0(x) # [B, 512, N]

        x, _ = torch.max(x, 2) # [B, 512]
        x = self.fc0(x) # [B, nCls]
        return x

