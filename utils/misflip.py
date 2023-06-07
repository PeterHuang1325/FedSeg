import numpy as np
import torch

def misflipping(labelist, n_classes, fliptype='comple'):
    
    '''convert to numpy'''
    labelist = labelist.data.cpu().numpy()
    mislabel_full = np.zeros_like(labelist)
    
    '''complement flipping'''
    if fliptype == 'comple':
        for i, lbl in enumerate(labelist):
            mislabel_full[i] = 9-lbl
            
    '''random flipping'''
    if fliptype == 'random':
        for i, lbl in enumerate(labelist):
            class_list =list(range(n_classes))
            class_list.remove(lbl)
            np.random.seed(100+i)
            mislabel_full[i] = np.random.choice(class_list)
            
    return torch.from_numpy(mislabel_full)