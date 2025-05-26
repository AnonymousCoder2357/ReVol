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
    """
    Computes dynamic mean (mu) and standard deviation (sigma) from input return sequences
    using a linear projection followed by LSTM and attention mechanism.
    """
    def __init__(self, hidden_size):
        super(self.__class__, self).__init__()
        self.linear=nn.Linear(4,hidden_size)
        self.lstm=nn.LSTM(hidden_size,hidden_size,1)

    def forward(self, x):
        """
        Compute the mean (mu) and standard deviation (sigma) of the return series using attention over LSTM outputs.

        param x: Input tensor of shape (B, T, F) or (B, N, T, F), where F must be 4.
        
        return: Tuple (mu, sigma) with shape (B, N) or (B,), depending on input shape.
        """
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
    """
    ReVol normalizes input return sequences based on learned mu and sigma,
    then performs prediction using a base forecasting model.
    """
    def __init__(self, mu_hidden, oc_ratio, window, device,fm):
        super().__init__()
        self.window=window
        self.oc_ratio=oc_ratio
        self.device=device
        self.model = fm
        self.mu_sigma = MuSigma(mu_hidden)
        self.sqrt_r=math.sqrt(oc_ratio)
    
    def forward(self, x, mask, with_mu=False):
        """
        Normalize input using computed mu and sigma, then make predictions with the underlying model.

        param x: Input tensor of shape (B, T, F) or (B, N, T, F), with F=4.
        param mask: Mask tensor for sequence modeling (e.g., attention mask or padding mask).
        param with_mu: Boolean flag to return mu and sigma along with prediction.

        return: Predicted values with optional mu and sigma, shape (B, T-1), or tuple (pred, mu, sigma) if with_mu=True.
        """
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
    """
    A simple LSTM-based forecasting model with a linear input projection and output prediction layer.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear=nn.Linear(4,hidden_size)
        self.lstm=nn.LSTM(hidden_size,hidden_size,1)
        self.pred=nn.Linear(hidden_size,1)

    def forward(self,x,mask):
        """
        Process the input sequence through LSTM and output predicted returns.

        param x: Input tensor of shape (B, N, T, F), where F=4.
        param mask: Mask tensor (not used in this basic LSTM model, kept for compatibility).

        return: Prediction tensor of shape (B, N).
        """
        x=x[:,:-1,:,:].contiguous()
        out=x.view(-1,x.size(2),x.size(3))
        out=self.linear(out) #b t h
        out=torch.tanh(out)
        out=out.transpose(0,1) #t b h
        out,_=self.lstm(out)[-1]
        out=out.transpose(0,1)
        return self.pred(out).view(x.size(0),x.size(1))