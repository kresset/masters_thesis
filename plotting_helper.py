#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:15:35 2019

@author: elisabeth
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def plot_input(r, c, x_data, y_data, figsize):
    f, ax = plt.subplots(r,c,figsize=figsize, sharey=True, dpi=300)
    plt.suptitle("Inputs to classify", size=20)
    params = [[-1, "#602ba0", "f1<f2"], [1, "#25681b", "f1>f2"]]
    for r in range(r):
        for i in range(c):
            ax[r,i].set_title(np.where(y_data.numpy()[:,-1]==params[r][0])[0][i])
            ax[r,i].plot(x_data[np.where(y_data.numpy()[:,-1]==params[r][0])[0][i],:].numpy(), c=params[r][1], label =params[r][2])
            ax[r,i].plot(y_data[np.where(y_data.numpy()[:,-1]==params[r][0])[0][i],:].numpy(), c=params[r][1], alpha = 0.5, label ='Decision')
            ax[r,i].set_label(params[r][2])
            ax[r,i].legend(loc=3)
        ax[r,0].set_ylabel("Fequency", size = 16)
    ax[r,2].set_xlabel("Time (ms)", size = 16)
    plt.ylim([-1.2, 2.0])
    plt.savefig("V_T Class Input to network")
    plt.show()

def performance_input(r, c, figsize, x, y, output, batches, s1, s2, dec_time, top):
    performance = np.sum(np.sign(output[:,dec_time])==y[:,dec_time])/batches

    f, ax = plt.subplots(r, c, figsize=figsize, dpi=300)
    plt.suptitle("Performance of the Network %.2f" %np.round(performance,2), size= 16)
    count = 0
    for j in range(r):
        for i in range(c):
            plt.subplots_adjust(hspace=3)
            ax[j,i].set_title('Label %d' %int(y[count,-1]))
            if float(x[count,s1]) > float(x[count,s2]):
                col = "#25681b"
                ax[j,i].plot(x[count,:], c=col, label = "f1: %.1f f2: %.1f" %(float(x[count,s1]),float(x[count,s2])))
                ax[j,i].plot(y[count,:], c=col, alpha= 0.7, label = "Decision")
                ax[j,i].plot(output[count,:], label = "Output")
                ax[j,i].set_ylim((-2.0,2.0))
                ax[j,i].legend()
            elif float(x[count,s1]) < float(x[count,s2]):
                col = "#602ba0"
                ax[j,i].plot(x[count,:], c=col, label = "f1: %.1f f2: %.1f" %(float(x[count,s1]),float(x[count,s2])))
                ax[j,i].plot(y[count,:], c=col, alpha= 0.7, label = "Decision")
                ax[j,i].plot(output[count,:], label = "Output")
                ax[j,i].set_ylim((-2.0,2.0))
                ax[j,i].legend()    
            count+=1
    plt.tight_layout()
    plt.subplots_adjust(top=top)
    plt.savefig("Output for Vibro-Tactile")
    plt.show()
  
def tuning_percentage(frac_sig_tuned, title, s1,s2,dur, dec_time, tune_time, error):
    plt.figure(figsize=(12,6), dpi=300)
    plt.title(title, size=16)
    if error==True:
        std = [frac_sig_tuned.mean(0) - np.std(frac_sig_tuned, axis=0), frac_sig_tuned.mean(0)+np.std(frac_sig_tuned, axis=0)]
        plt.fill_between(np.linspace(0,80,80), std[0],std[1], alpha=0.2)
        plt.plot(frac_sig_tuned.mean(0), label="Fraction of Neurons")
    else:
        plt.plot(frac_sig_tuned, label="Fraction of Neurons")
    plt.fill_between((s1,s1+dur),(1.0,1.0), alpha = 0.5, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True, label="Stim1, Stim2")
    plt.fill_between((s2,s2+dur),(1.0,1.0), alpha = 0.5, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True)
    plt.axvline(dec_time, color='k', linestyle=':', linewidth=1, label="Decision 10 ms Delay")
    plt.axvline(tune_time, color='k', linestyle='-.', linewidth=1, label="Tuned Time")
    plt.legend(loc=2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Fraction of Significantly Tuned a1")
    plt.savefig(title)
    plt.show()
    
def rastor_plot(r, c, spikes, data, s1, s2, steps, nb_rec, title):
    fig, ax = plt.subplots(r,c, figsize=(16,10), dpi=300)
    plt.suptitle(title, size = 16)
    count = 0
    for j in range(r):
        for i in range(c):
            ax[j,i].set_title("S1: %.2f S2: %.2f" %(np.round(data[10*count,s1],2),np.round(data[(10*count)+1,s2],2)))
            ax[j,i].imshow(spikes[10*count,:,:].T, cmap=plt.cm.gray_r, interpolation='nearest', aspect='auto')
            ax[j,i].grid(False)
            ax[1,i].set_xlabel("Time (ms)")
            ax[j,0].set_ylabel("Neurons")   
            count +=1
    plt.savefig(title)
    plt.show()
    
def rastor_stim_same(r,c, spikes, data, s1, s2):
    fig, ax = plt.subplots(r,c, figsize=(16,10), dpi=300)
    plt.suptitle("Same neural spike activity for frequency pair", size = 16)
    count = 0
    
    for j in range(r):
        for i in range(c):
            ax[j,i].set_title("S1: %.2f S2: %.2f" %(np.round(data[count,s1],2),np.round(data[count,s2],2)))
            ax[j,i].imshow(spikes[count,:,:].T, cmap=plt.cm.gray_r, interpolation='nearest', aspect='auto')
            ax[j,i].grid(False)
            ax[1,i].set_xlabel("Time (ms)")
            ax[j,0].set_ylabel("Neurons")
            ax[j,i].grid(False)
            count +=1
    plt.savefig("10 trials same f")
    plt.show()

def freq_stim(r,c,spikes, data, s1,s2):
    pop_rate = np.asarray([spikes[count*10:(count+1)*10,:,:].mean(0).mean(1) for count in range(10)])*1000
    fig, ax = plt.subplots(r,c, figsize=(16,10), sharey=True, dpi=300)
    plt.suptitle("Firing Rate of Population", size = 16)
    count = 0
    for j in range(r):
        for i in range(c):
            ax[j,i].set_title("S1: %.2f S2: %.2f" %(np.round(data[10*count,s1],2),np.round(data[(10*count)+1,s2],2)))
            ax[j,i].plot(pop_rate[count,:])
            ax[j,i].fill_between((10,20),(np.max(pop_rate),np.max(pop_rate)), alpha = 0.5, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True, label="Stim1, Stim2")
            ax[j,i].fill_between((50,60),(np.max(pop_rate),np.max(pop_rate)), alpha = 0.5, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True)
            ax[j,i].grid(False)
            ax[1,i].set_xlabel("Time (ms)")
            ax[j,0].set_ylabel("Population Firing Rate (Hz)")
            ax[j,i].grid(False)
            count +=1
    plt.savefig("Average firing rate of population for frequencies")
    plt.show()
    
def output_plot(samples_yes, samples_no, avg_yes, avg_no, num):
    plt.figure(figsize = (16,8), dpi=300)
    plt.suptitle("Model Output for different trials", size = 16)
    plt.plot(avg_no, label = "avg f1<f2", linewidth=2, c ="r")
    plt.plot(avg_yes, label = "avg f2<f1", linewidth=2, c = "b")
    for i in range(num):
        plt.plot(samples_no[i,:], linewidth=0.5, c ="r")
        plt.plot(samples_yes[i,:], linewidth=0.5, c ="b")
    plt.fill_between((10,20),(-1,-1), alpha = 0.7, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True)
    plt.fill_between((10,20),(1,1), alpha = 0.7, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True)
    plt.fill_between((50,60),(-1,-1), alpha = 0.7, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True)
    plt.fill_between((50,60),(1,1), alpha = 0.7, edgecolor='#c4c4ce', facecolor ='#c4c4ce', interpolate=True)
    plt.axvline(70, color='k', linestyle=':', linewidth=1, label="Decision 10 ms Delay")
    plt.axhline(0, color='k', linestyle='-', linewidth=3, label="Decision Divide")
    plt.xlabel("Time (ms)")
    plt.ylabel("Output")
    plt.legend()
    plt.savefig("Output plot")
    plt.show()
    