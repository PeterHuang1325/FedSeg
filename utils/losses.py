import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utils.gam_select import *


def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    #bce_loss = nn.BCEWithLogitsLoss()(pred, label)
    return dice_loss + bce_loss

def gamma_dice_loss(pred, label, gamma=5e-5): #1e-4
    '''
    gamma selection: positive number, min: 1e-7
    '''
    pos = torch.sigmoid((1+gamma)*pred)
    neg = 1 - pos
    power = gamma/(1+gamma)
    #gamma_loss = (1/gamma)*torch.meanP(label*(1-(pos**power)) + (1-label)*(1-(neg**power)))
    gamma_loss = torch.mean(label*((1-(pos**power))/gamma) + (1-label)*((1-(neg**power))/gamma))
    dice_loss = dice_coef_loss(pred, label)
    return gamma_loss + dice_loss
'''
def auto_gamma_dice_loss(pred, label, gamma=1e-4):
    #gamma selection: positive number, min: 1e-6
    pos = torch.sigmoid((1+gamma)*pred)
    neg = 1 - pos
    power = gamma/(1+gamma)
    #gamma_loss = (1/gamma)*torch.mean(label*(1-(pos**power)) + (1-label)*(1-(neg**power)))
    score = label*((1-(pos**power))/gamma) + (1-label)*((1-(neg**power))/gamma)
    gamma_loss = torch.mean(score)
    dice_loss = dice_coef_loss(pred, label)
    tot_loss = gamma_loss + dice_loss
    return tot_loss, score

'''
def auto_gamma_dice_loss(pred, label, gamma=1e-4):
    pos = torch.sigmoid((1+gamma)*pred)
    neg = 1 - pos
    power = gamma/(1+gamma)
    #gamma_loss = (1/gamma)*torch.mean(label*(1-(pos**power)) + (1-label)*(1-(neg**power)))
    score = label*((1-(pos**power))/gamma) + (1-label)*((1-(neg**power))/gamma)
    '''gamma selection for this batch, used for computing mean gamma for next FL round'''
    gam_update = gam_logit_select(score, pred, label) 
    gamma_loss = torch.mean(score)
    dice_loss = dice_coef_loss(pred, label)
    tot_loss = gamma_loss + dice_loss
    return tot_loss, gam_update

    
def combo_loss(pred, label, alpha=0.8):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    #bce_loss = nn.BCEWithLogitsLoss()(pred, label)
    return alpha*bce_loss + (1-alpha)*dice_loss

def mse_loss(pred, label):
    return nn.MSELoss()(pred,label)

def mse_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    mse_loss = nn.MSELoss()(pred, label)
    #bce_loss = nn.BCEWithLogitsLoss()(pred, label)
    return dice_loss + mse_loss

#focal loss: let the model not pass sigmoid layer
def focal_loss(pred, label):
    #focal loss
    focal_loss = torchvision.ops.sigmoid_focal_loss(pred, label, alpha=0.75, gamma=2, reduction = 'mean')
    return focal_loss

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]
    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
        
def tversky_loss(y_pred, y_true, delta = 0.6, smooth = 0.000001):
    axis = identify_axis(y_true.size())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    '''
    tp = torch.sum(y_true * y_pred, dim=axis)
    fn = torch.sum(y_true * (1-y_pred), dim=axis)
    fp = torch.sum((1-y_true) * y_pred, dim=axis)
    '''
    
    tp = (y_pred*y_true).sum()
    fn = ((1-y_pred)*y_true).sum()
    fp = (y_pred*(1-y_true)).sum()
    
    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Average class scores
    tversky_loss = torch.mean(1-tversky_class)

    return tversky_loss

def cross_entropy(pred, label, n_classes):
    return torch.nn.CrossEntropyLoss()(pred, label)

'''multi-class classification problem'''
def gamma_logit_loss(pred, label, n_classes, gamma=5e-5): #0.1, 1e-4, 0.5
    #softmax
    pos = ((1+gamma)*pred).softmax(dim=1)
    power = gamma/(1+gamma)
    prob = (1-(pos**power))/gamma
    '''one hot encoding'''
    one_lbl = F.one_hot(label, num_classes=n_classes)
    '''compute gamma logistic loss'''
    score = (prob*one_lbl).sum(dim=1)
    gamma_loss = torch.mean(score)

    return gamma_loss

'''
FedDG
'''
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)
    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
