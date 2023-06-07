import torch
import numpy as np
import torch.nn as nn
import torchvision

def gam_select(points, prevs, rho=0.5):
    eps = 1e-5
    x_median = np.median(points,axis=0)
    #sig_sq = np.mean((points - np.mean(points, axis=0))**2)
    dist = [np.square(np.linalg.norm(p - prevs)) for p in points] #Kx1
    dist_med = np.median(dist)
    gamma = (-2*np.log(rho)) / (dist_med+eps)
    gamma_clip = np.clip(gamma, 1e-4, 10) #clipping
    return gamma_clip

'''
def gam_logit_select(points, rho=0.5):
    eps = 1e-5
    points = points.data.cpu().numpy() #(3005, 1, 256, 256)
    meds = np.median(points)
    dist = np.square(np.linalg.norm(points - meds))/(points.shape[-2]*points.shape[-1])
    gamma = (-2*np.log(rho)) / (dist+eps)
    gamma_clip = np.clip(gamma, 1e-4, 1) #clipping
    return gamma_clip

'''

def gam_logit_select(points, pred, label, rho=0.5):
    #points shape: (batch_size, 1, 256, 256)
    eps = 1e-5
    #points, pred, label = points.data.cpu().numpy(), pred.data.cpu().numpy(), label.data.cpu().numpy()
    '''get median position index'''
    slice_points = torch.mean(points.reshape((points.shape[0],-1)), axis=-1) #(16, 256x256)
    med_idx = torch.argsort(slice_points)[len(slice_points)//2] #median index
    #print(pred.shape, med_idx)
    '''compute pos and neg'''
    logit = torch.sigmoid(pred[med_idx]) #compute logit for median
    '''compute positive and negative'''
    #pos = (1-rho)*torch.log(logit) #(rho-1)*(-1)
    pos = (rho-1)/(torch.log(logit)+eps)
    #neg = -(1-rho)*torch.log(1+torch.exp(pred[med_idx]))#(rho-1)
    neg = (1-rho)/(torch.log(1+torch.exp(pred[med_idx]))+eps)
    
    gamma = torch.mean(label[med_idx]*pos+(1-label[med_idx])*neg) #add minus for argmax to argmin
    gam_clip = torch.clamp(gamma, 1e-4, 1).data.cpu().numpy()
    return gam_clip