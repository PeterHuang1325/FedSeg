import numpy as np
import albumentations as A
import random
import torch

'''ellipse mask mislabel'''
def ellipse_mask(mask):
    #np.random.seed(1)
    rands = np.random.randint(1, 5, (2,))
    center = np.random.randint(-5, 5, (2,))
    
    x0 = center[0]; a = rands[0]  # x center, half width                                       
    y0 = center[1]; b = rands[1]  # y center, half height
    x = np.linspace(-10, 10, mask.shape[-1])  # x values of interest
    y = np.linspace(-10, 10, mask.shape[-1])[:,None]  # y values of interest, as a "column" array
    ellipse = ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1  # True for points inside the ellipse
    ellipse = ellipse.astype('int').reshape(mask.shape)
    return ellipse

'''
mislabeling 
'''
def mislabeling(masklist, epc):
    '''set seed'''
    random.seed(100+epc)
    np.random.seed(100+epc)

    '''convert to numpy'''
    masklist = masklist.data.cpu().numpy()
    '''split idx for zoom out flipping or ellipse mislabel'''
    idx = masklist.shape[0] // 2
    '''create mismask array'''
    mismask_full = np.zeros_like(masklist)
    
    for i, mask in enumerate(masklist):
        #first half
        if i <= idx:
            '''affine transformation'''
            mismask = A.Affine(scale=0.5, rotate=180, p=1)(image=mask)['image']
        #second half
        else:
            '''ellipse mislabeling'''
            mismask = ellipse_mask(mask)
        mismask_full[i] = mismask
    return torch.from_numpy(mismask_full)