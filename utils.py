# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:22:07 2019
@author: joram
"""

import numpy as np
import torch
dtype = torch.float
    
class SuperSpike(torch.autograd.Function):
    """ SuperSpike surrogate gradient """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        spk = torch.where(x > 0, torch.tensor(1, dtype=dtype), torch.tensor(0, dtype=dtype))
        return spk
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * 1.0 / (1.0 + 100 * torch.abs(x)) ** 2
        return grad_input
        
        
class GradClipper(object):
    """ Clips gradients between hi and lo
        Example:
        clipper = GradClipper(-1,1) # init
        model.apply(clipper)
    """
    
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        
    def __call__(self, module):
        for p in module.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(self.lo, self.hi)
