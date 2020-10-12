#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 12:24:49 2019

@author: elisabeth
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

# Setting the device and type will have to change for GPU
dtype = torch.float32
device = torch.device("cpu")

# Network Dynamics in miliseconds
time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3
#tau_ref = 2e-3
alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))

class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SuperSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SuperSpike.apply

def r_snn(inputs, w_ff, w_rec, w_out, batches, nb_steps, nb_rec, nb_out):
    syn = torch.zeros((batches, nb_rec), device=device, dtype=dtype)
    mem = torch.zeros((batches, nb_rec), device=device, dtype=dtype)
    spk = torch.zeros((batches, nb_rec), device=device, dtype=dtype)
    h2 = torch.zeros((batches, nb_out), device=device, dtype=dtype)
    
    mem_rec = [mem]
    spk_rec = [mem]
    out_rec = [h2]
    rst = 0
    
    ff = torch.einsum("abc,cd->abd", (inputs, w_ff))
    # Compute hidden layer activity
    for t in range(nb_steps-1):
            
            ref = (rst>0)
            mem[ref] = 0

            mthr = mem-1.0

            out = spike_fn(mthr)
            rst = torch.zeros_like(mem)
            c   = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]
            
            rec = torch.mm(out, w_rec)
            new_syn = alpha*syn + rec + ff[:,t,:]
            new_mem = beta*mem + syn - rst
            
            mem = new_mem
            syn = new_syn

            mem_rec.append(mem)
            spk_rec.append(out)
    
        # Readout layer

            h2 = torch.mm(out, w_out)
            out_rec.append(h2)
        
    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)
    out_rec = torch.stack(out_rec,dim=1)

    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs