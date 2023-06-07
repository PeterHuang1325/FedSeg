import numpy as np

def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if (pred.sum() == 0) and (label.sum() == 0):
        return 1.
    return intersection / union

def iou_metric(pred, label):
    intersection = np.sum(pred * label)
    union = np.sum(pred) + np.sum(label) - intersection
    if (np.sum(pred) == 0) and (np.sum(label) == 0):
        return 1 
    return intersection / union

'''accuracy'''
def acc_score(pred, label):
    return np.mean(pred == label)