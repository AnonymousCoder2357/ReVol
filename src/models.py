"""
The PyTorch implementation of ReVol.
"""
import torch
from torch import nn
from torch.nn import functional as func
from torch.nn.modules.activation import MultiheadAttention
import torch.nn.functional as F
import math

    
class MuSigma(nn.Module):
    def __init__(self, hidden_size):
        super(self.__class__, self).__init__()
        self.linear=nn.Linear(4,hidden_size)
        self.lstm=nn.LSTM(hidden_size,hidden_size,1)

    def forward(self, x):
        x_size=x.size()
        size_4=False
        if len(x_size)==4:
            x=x.view(x_size[0]*x_size[1],x_size[2],x_size[3])
            size_4=True

        out=self.linear(x)
        out=torch.tanh(out)


        out,_=self.lstm(out.transpose(0,1))
        attn=torch.bmm(out.transpose(0,1),out[-1].unsqueeze(-1)).squeeze(2)
        attn=torch.softmax(attn,dim=1)

        mu=torch.sum(attn*x[:,:,3],dim=1)
        sigma=torch.sqrt(torch.sum(attn*(x[:,:,3]-mu.unsqueeze(1))**2,dim=1))

        if size_4:
            return mu.view(x_size[0],x_size[1]),sigma.view(x_size[0],x_size[1])
        return mu,sigma
    

class ReVol(nn.Module):
    def __init__(self, mu_hidden, oc_ratio, window, device,fm):
        super().__init__()
        self.window=window
        self.oc_ratio=oc_ratio
        self.device=device
        self.model = fm
        self.mu_sigma = MuSigma(mu_hidden)
        self.sqrt_r=math.sqrt(oc_ratio)
    
    def forward(self, x, mask, with_mu=False):
        mu,sigma=self.mu_sigma(x)

        new_shape = list(x.shape)
        new_shape[-2] = self.window
        x_new=torch.empty(*new_shape).to(self.device)

        mu,sigma=mu.unsqueeze(-1),sigma.unsqueeze(-1)
        x_new[...,3]=(x[...,-self.window:,3]-mu)/sigma
        x_new[...,0]=(x[...,-self.window:,0]-mu*self.oc_ratio)/sigma/self.sqrt_r
        x_new[...,1:3]=x[...,-self.window:,1:3]/sigma.unsqueeze(-1)

        mu,sigma=mu.squeeze(-1),sigma.squeeze(-1)
        if with_mu:
            return self.model(x_new,mask)*sigma[:,:-1]+mu[:,:-1],mu,sigma
        else:
            return self.model(x_new,mask)*sigma[:,:-1]+mu[:,:-1]
        
class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear=nn.Linear(4,hidden_size)
        self.lstm=nn.LSTM(hidden_size,hidden_size,1)
        self.pred=nn.Linear(hidden_size,1)

    def forward(self,x,mask):
        x=x[:,:-1,:,:].contiguous()
        out=x.view(-1,x.size(2),x.size(3))
        out=self.linear(out) #b t h
        out=torch.tanh(out)
        out=out.transpose(0,1) #t b h
        out,_=self.lstm(out)[-1]
        out=out.transpose(0,1)
        return self.pred(out).view(x.size(0),x.size(1))