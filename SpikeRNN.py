#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:23:40 2019

@author: elisabeth
"""

import torch
import numpy as np
dtype = torch.float32
device = torch.device("cpu") #"cuda:1"
import helper_funcs
from tqdm import tqdm
import torch.nn as nn

class SpikeRNN(torch.nn.Module):
    def __init__(self, nb_in , nb_rec , nb_out , batches , nb_steps, noise_scale, time_step = 1e-3, tau_mem = 10e-3, tau_syn = 5e-3, dale = True):
        super(SpikeRNN, self).__init__()
        self.nb_in = nb_in
        self.nb_rec = nb_rec
        self.nb_out = nb_out
        self.time_step = time_step
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.nb_steps = nb_steps
        self.batches = batches
        self.noise_scale = noise_scale
        ### Weights
        self. alpha = float(np.exp(-self.time_step/self.tau_syn))
        self.beta = float(np.exp(-self.time_step/self.tau_mem))
        weight_scale = 7*(1.0-self.beta) # this should give us some spikes to begin with from friedemann
        
        #clamp_w_ff = np.random.choice([0,1,-1], p = [0.7, 0.15,0.15], size = (1, nb_rec))
        #self.w_ff = torch.tensor(clamp_w_ff*np.random.uniform(-1,1,100), dtype=dtype, requires_grad=True)
        
        self.w_ff = torch.empty((self.nb_in, self.nb_rec),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w_ff, mean=0.0, std=weight_scale/np.sqrt(self.nb_in))
        
        self.w_rec = torch.empty((self.nb_rec, self.nb_rec),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w_rec, mean=0.0, std=weight_scale/np.sqrt(self.nb_rec))
    
        self.w_out = torch.empty((self.nb_rec, self.nb_out),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w_out, mean=0.0, std=weight_scale/np.sqrt(self.nb_rec))
        
        self.dale = dale
        if dale:
    	    # Create random sign matrix
            I_prob = 0.5
            self.D_sign = torch.diag(torch.tensor(np.random.choice([-1,1], p = [I_prob, 1-I_prob], size=self.nb_rec), 
                                    dtype=dtype, requires_grad=False))

        # here we overwrite our naive spike function by the "SuperSpike" nonlinearity which implements a surrogate gradient
        
    def forward(self, inputs):
        noise = torch.tensor(np.random.normal(scale=self.noise_scale, 
                            size = (self.batches, self.nb_steps, self.nb_rec)),
                            dtype=dtype)
        
        syn = torch.zeros((self.batches, self.nb_rec), device=device, dtype=dtype)
        mem = torch.zeros((self.batches, self.nb_rec), device=device, dtype=dtype)
        h2 = torch.zeros((self.batches, self.nb_out), device=device, dtype=dtype)
        ref_store = [torch.zeros((self.batches, self.nb_rec), device=device, dtype=dtype)]
        
        mem_rec = [mem]
        spk_rec = [mem]
        out_rec = [h2]
        rst = 0
        
        if self.dale:
	    # Effective matrices: matrix + sing constrain
            # Input matrix: no Dale
            w_rec = torch.abs(self.w_rec)
            w_rec_d = torch.mm(self.D_sign, w_rec)
            w_out = torch.abs(self.w_out)
            w_out_d = torch.mm(self.D_sign, w_out)
            w_ff_d = self.w_ff
        
        else:
	    # No sign constraints
            self.w_rec = self.w_rec
            self.w_out = self.w_out
            self.w_ff = self.w_ff
        
        in_ff = torch.einsum("abc,cd->abd", (inputs, w_ff_d))
        for t in range(self.nb_steps-1):
        
        
            ref = (rst>0)
            ref2 = (ref_store[t-1]>0)
            mem[ref] = 0
            mem[ref2] = 0
    
            mthr = mem-1.0
    
            out = helper_funcs.SuperSpike.apply(mthr)
            rst = torch.zeros_like(mem)
            c   = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]
            
            rec = torch.mm(out, w_rec_d)
            new_syn = self.alpha*syn + rec + in_ff[:,t,:] + noise[:,t]
            new_mem = self.beta*mem + syn - rst
            
            mem = new_mem
            syn = new_syn
    
            mem_rec.append(mem)
            spk_rec.append(out)
            ref_store.append(rst)
            
            # Readout layer
            h2 = torch.mm(out, w_out_d)
            out_rec.append(h2)
             
        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)
        out_rec = torch.stack(out_rec,dim=1)
        w_ff = self.w_ff
        other_recs = [mem_rec, spk_rec]
        return out_rec, spk_rec, w_rec_d, w_out_d, w_ff
    
def av_tuning_test(spike_network, s1, s2, dec, iters, epochs, spike_scale, L2_scale):
    f_range = np.linspace(0.2,1.8,7)
    f_dec = np.array((-1,1))
    dec_total = np.empty((iters,spike_network.nb_steps))
    s1_total = np.empty((iters,spike_network.nb_steps))
    s2_total = np.empty((iters,spike_network.nb_steps))
    xf, yf = helper_funcs.create_data_nonrandom(100, spike_network.nb_steps, 0.2, 1.8, 10, 50, 10, 10)

    for i in tqdm(range(iters)):
        spike_network = SpikeRNN(1,100,1,60,80,noise_scale=0)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam([spike_network.w_ff, spike_network.w_rec, spike_network.w_out], lr=1e-3, betas=(0.9,0.999))

        for e in range(epochs):

            x,y = helper_funcs.create_data(spike_network.batches, spike_network.nb_steps, 0.2, 1.8, 10, 50, 10, 10)
            output ,out_dyn,_,_,_ = spike_network(x)

            # Regularizing over total number of spikes from all batches, time steps and neurons
            #reg_loss = spike_scale*max(out_dyn[1].sum(0).sum(0))

            # L2 Regularization over weights.
            reg_loss = L2_scale*(0.5*(torch.sum(spike_network.w_rec)**2 + torch.sum(spike_network.w_out)**2 + torch.sum(spike_network.w_ff)**2))

            loss = loss_fn(output[:,70:], y[:,70:]) + reg_loss

            if e % int(epochs / 10) == 0:
                print(e+1, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        spike_network.batches = 100
        out_freq_spikes, out_freq_dyn, _,_,_ = spike_network(xf)

        _, s1_total[i,:] = helper_funcs.tuning_calc(out_freq_dyn, xf, spike_network.nb_steps, spike_network.nb_rec, f_range, 15)        
        _, s2_total[i,:] = helper_funcs.tuning_calc(out_freq_dyn, xf, spike_network.nb_steps, spike_network.nb_rec, f_range, 55)
        _, dec_total[i,:] = helper_funcs.tuning_calc(out_freq_dyn, yf, spike_network.nb_steps, spike_network.nb_rec, f_dec, 70)
    return s1_total, s2_total, dec_total
