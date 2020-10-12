#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:09:35 2019

@author: elisabeth
"""
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter1d
dtype = torch.float32
from scipy.stats import linregress


class SuperSpike(torch.autograd.Function):
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike.scale*torch.abs(input)+1.0)**2
        return grad
    

def create_data(batches, steps, fl, fh, s1, s2, dur, dec_delay):
    f_range = np.hstack((np.vstack((np.linspace(fl, fh, 7)[0:5],np.linspace(fl, fh, 7)[2:7])),
                         np.vstack((np.linspace(fl, fh, 7)[2:7], np.linspace(fl, fh, 7)[0:5]))))
    x = np.zeros((batches, steps, 1))
    y = np.zeros((batches, steps, 1))
    
    for i in range(batches):
        fs = np.random.choice(np.arange(0,10))
        x[i, s1:s1+dur] = f_range[0][fs]
        x[i, s2:s2+dur] = f_range[1][fs]
        
        if f_range[0][fs] > f_range[1][fs]:
            y[i, s2+dur+dec_delay:] = 1
        else:
            y[i, s2+dur+dec_delay:] = -1

    # To Tensor
    x = torch.tensor(x, dtype=dtype)
    y = torch.tensor(y, dtype=dtype)
    return x , y

def create_data_nonrandom(batches, steps, fl, fh, s1, s2, dur, dec_delay):
    f_range = np.hstack((np.vstack((np.linspace(fl, fh, 7)[0:5],np.linspace(fl, fh, 7)[2:7])),
                         np.vstack((np.linspace(fl, fh, 7)[2:7], np.linspace(fl, fh, 7)[0:5]))))
    x = np.zeros((batches, steps, 1))
    y = np.zeros((batches, steps, 1))
    count =0
    for j in range(10):
        x[10*count:10*(count+1), s1:s1+dur] = f_range[0,j]
        x[10*count:10*(count+1), s2:s2+dur] = f_range[1,j]
        
        if f_range[0,j] > f_range[1,j]:
            y[10*count:10*(count+1), s2+dur+dec_delay:] = 1
        else:
            y[10*count:10*(count+1), s2+dur+dec_delay:] = -1
        count +=1
    # To Tensor
    x = torch.tensor(x, dtype=dtype)
    y = torch.tensor(y, dtype=dtype)
    return x , y

def model_output(batches, steps, fl, fh, s1, s2, dur, dec_delay):
    x = np.zeros((batches, steps, 1))
    y = np.zeros((batches, steps, 1))

    for i in range(batches):
        fs = np.random.uniform(fl,fh,2)
        x[i, s1:s1+dur] = fs[0]
        x[i, s2:s2+dur] = fs[1]

        if fs[0] > fs[1]:
            y[i, s2+dur+dec_delay:] = 1
        else:
            y[i, s2+dur+dec_delay:] = -1

    # To Tensor
    x = torch.tensor(x, dtype=dtype)
    y = torch.tensor(y, dtype=dtype)
    return x,y 

def create_data_uniform(batches, steps, fl, fh, s1, s2, dur, dec_delay):
    x = np.zeros((batches, steps, 1))
    y = np.zeros((batches, steps, 1))

    for i in range(batches):
        fs = np.round(np.random.uniform(fl, fh ,1), 1)
        x[i, s1:s1+dur] = fs
        fs2 = np.round(np.random.uniform(fs-0.2, fs+0.2, 1),1)
        x[i, s2:s2+dur] = fs2

        if fs >= fs2:
            y[i, s2+dur+dec_delay:] = 1
        else:
            y[i, s2+dur+dec_delay:] = -1
        
    # To Tensor
    x = torch.tensor(x, dtype=dtype)
    y = torch.tensor(y, dtype=dtype)
    
    return x,y

def tuning_calc(out_dyn, data, steps, nb_rec, f_range, tune_time):
    """
    Func: Calculates the tuning of neurons in relation to stimulus at time=tune_time
    
    Input: out_dyn, data, steps, f_range, tune_time
        
    Return: slopes, frac_sig
    """
    
    freq_tune = np.empty((len(f_range), steps, nb_rec))
    for j in range(nb_rec):
        for i,k in enumerate(f_range):
            f_test = np.where(data[:,tune_time,:] == k)[0]
            freq_tune[i,:,j] = out_dyn[1].detach().numpy()[f_test,:,j].mean(axis=0)
    
    slopes = np.empty((steps,nb_rec,4))
    for j in range(nb_rec):
        for i in range(steps):
            slopes[i,j,0], slopes[i,j,1],slopes[i,j,3], slopes[i,j,2], _ = linregress((f_range, freq_tune[:,i,j]))
            
    sig_p = np.sort(np.hstack((np.array([np.where(slopes[:,i,2]<0.05)[0] for i in range(nb_rec)]))))
    frac_sig = np.array([np.sum(sig_p==i)/nb_rec for i in range(steps)])
    return slopes, frac_sig

def s1_theories(spikes, data, f_range, nb_rec):
    wm_count = np.empty((len(f_range)))
    freq_wm = np.empty((len(f_range)))
    total_spike = np.empty((len(f_range)))
    for j in range(nb_rec):
        for i,k in enumerate(f_range):
            f_test = np.where(data[:,15,:] == k)[0]
            total_spike[i] = (spikes[f_test,:,:].mean())
            wm_count[i] = (spikes[f_test,23:50,:].sum(1) == 0.0).sum(1).mean()
            freq_wm[i] = (spikes[f_test,23:50,:].sum(1)).mean(0).mean()
    return wm_count, freq_wm, total_spike

