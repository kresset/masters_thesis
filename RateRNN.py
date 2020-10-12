# -*- coding: utf-8 -*-
"""
Pytorch implementation of Rate network
with 
Created on Tue Apr  9 15:55:45 2019
@author: joram
"""

import torch
import numpy as np
dtype = torch.float
#from utils import SuperSpike
import helper_funcs
import torch.nn as nn
from tqdm import tqdm

dtype = torch.float

class RateRNN(torch.nn.Module):
    def __init__(self, n_in, n_rec, n_out, batches, nb_steps, noise_scale, g = 10,
                 tau = 10e-3, dt = 1e-3, dale = True):
        super(RateRNN, self).__init__()
        
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.batches = batches
        self.nb_steps = nb_steps
        self.dt = torch.tensor(dt, dtype=dtype)
        self.tau = torch.tensor(tau, dtype=dtype)
        self.phi = lambda x: 1 / (1 + torch.exp(-x)) # Non-negative rates
        self.noise_scale = noise_scale
	# Uncomment for unconstrained 'rates'
	#self.phi = lambda x: torch.tanh(x)
        
        # Initialize weights
        rng = np.random.RandomState(None)
        self.w_in = torch.nn.Parameter(torch.tensor(rng.normal(
            scale = 1.5 / np.sqrt(n_in), size = (n_in, n_rec)),
                                dtype=dtype, requires_grad = False))
        # Recurrent matrix: sparse init
        w_rec = rng.normal(scale = g / np.sqrt(n_rec), size = (n_rec, n_rec))
        self.w_rec = torch.nn.Parameter(torch.tensor(w_rec,
                                dtype=dtype, requires_grad = True))
        self.w_out = torch.nn.Parameter(torch.tensor(rng.normal(
            scale = 1.5 / np.sqrt(n_rec), size = (n_rec, n_out)),
                                dtype=dtype, requires_grad = True))
        
        self.dale = dale
        if dale:
	    # Create random sign matrix
            self.D_sign = torch.diag(torch.tensor(np.random.choice([-1,1], 
                                            p = [0.5, 1-0.5], size=n_rec), 
                                             dtype=dtype, requires_grad=False))
                
    def forward(self, inputs):
        """
        Forward pass
        Inputs:
            inputs: of size (batch_size, n_in, time_steps)
        Returns:
            x: network states, of size (batch_size, time_steps, n_rec)
            r: network rates, of size (batch_size, time_steps, n_rec)
            o: network outputs, of size (batch_size, time_steps, n_out)
        """
        batch_size, time_steps = inputs.shape[:2]
        x_rec = []
        x = torch.zeros((batch_size, self.n_rec), dtype=dtype)
        r = self.phi(x)
        noise = torch.tensor(
            np.random.normal(scale=self.noise_scale, 
                             size = (batch_size, time_steps, self.n_rec)),
                             dtype=dtype)
             
        if self.dale:
            w_rec = torch.abs(self.w_rec)
            w_eff = torch.mm(self.D_sign, w_rec)
            w_out = torch.abs(self.w_out)
            w_out_eff = torch.mm(self.D_sign, w_out)
            
        else:
	    # No sign constraints
            w_eff = self.w_rec
            w_out_eff = self.w_out

        for t in range(time_steps):
            x = (1 - self.dt / self.tau) * x + self.dt / self.tau * r @ w_eff + self.dt / self.tau * (inputs[:,t] @ self.w_in)
            x = x + noise[:,t] # Add noise
            r = self.phi(x)
            x_rec.append(x)
            
        x_rec = torch.stack(x_rec, dim = 1)
        r_rec = self.phi(x_rec)
        o_rec = torch.matmul(r_rec, w_out_eff)
        w_in = self.w_in
        return x_rec, r_rec, o_rec, w_eff, w_out_eff, w_in

def av_tuning_test(rate_network, s1, s2, dec, iters, epochs, noise_scale, L2_scale):
    f_range = np.linspace(0.2,1.8,7)
    f_dec = np.array((-1,1))
    dec_total = np.empty((iters,rate_network.nb_steps))
    s1_total = np.empty((iters,rate_network.nb_steps))
    s2_total = np.empty((iters,rate_network.nb_steps))
    xf, yf = helper_funcs.create_data_nonrandom(100, rate_network.nb_steps, 0.2, 1.8, 10, 50, 10, 10)

    for i in tqdm(range(iters)):
        rate_network = RateRNN(1,500,1, 60, 80, noise_scale, dale=True)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam([rate_network.w_in, rate_network.w_rec, rate_network.w_out], lr=1e-3, betas=(0.9,0.999))
        
        for e in range(epochs):
            
            x,y = helper_funcs.create_data(rate_network.batches, rate_network.nb_steps, 0.2, 1.8, 10, 50, 10, 10)
            x_rec, r_rec, o_rec, w_rec, w_out,_  = rate_network(x)
        
            # L2 Regularization over weights.
            reg_loss = 2e-6*(0.5*(torch.sum(w_rec)**2 + torch.sum(w_out)**2 + torch.sum(rate_network.w_in)**2))
            
            loss = loss_fn(o_rec[:,70:], y[:,70:]) + reg_loss
            
            if e % int(epochs / 1) == 0:
                print(e+1, loss.item())
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        rate_network.batches = 100
        x_dyn, rec_dyn, out_dyn, _, _,_ = rate_network(xf)
        out_freq_dyn = [x_dyn,rec_dyn]
        _, s1_total[i,:] = helper_funcs.tuning_calc(out_freq_dyn, xf, rate_network.nb_steps, rate_network.n_rec, f_range, 15)        
        _, s2_total[i,:] = helper_funcs.tuning_calc(out_freq_dyn, xf, rate_network.nb_steps, rate_network.n_rec, f_range, 55)
        _, dec_total[i,:] = helper_funcs.tuning_calc(out_freq_dyn, yf, rate_network.nb_steps, rate_network.n_rec, f_dec, 70)
    return s1_total, s2_total, dec_total
