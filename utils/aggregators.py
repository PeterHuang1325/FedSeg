import numpy as np

#we might not use np since the cpu computing is surprising large
#need to use tf or cp
import wquantiles as w

from functools import partial
from scipy.linalg import svd

def mean(points, weights=None):
    return np.average(points, axis=0, weights=weights)    


def std(points, weights=None):
    mu = mean(points, weights)
    return np.sqrt(mean(np.subtract(points, mu)**2, weights))


def cov(points, weights=None):
    """
    points.shape = (m, p=[...])
    cov.shape should be (p, p)
    """
    if weights is None:
        return np.cov(np.transpose([p.reshape([-1]) for p in points]))
    return np.cov(np.transpose([p.reshape([-1]) for p in points]), aweights = weights)


def median(points, weights = None):
    return quantile(points, weights, 0.5)


def coordinatewise(fn, points, weights=None):
    if len(points) == 1:
        return fn(points, weights)
    shape = np.shape(points[0])
    points = np.transpose([np.reshape(p, [-1]) for p in points])
    return np.transpose(list(map(lambda x: fn(x, weights), points))).reshape(shape)
    #return np.transpose([fn(v, weights) for v in points]).reshape(shape)


# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def quantile(points, weights = None, quantile = 0.5):
    if weights is None:
        return np.quantile(points, quantile, axis=0).astype(np.float32)
    return coordinatewise(partial(w.quantile_1D, quantile=quantile), points, weights)
def ext_remove(points, weights=None, beta=0.1):
    """
    keep the data from quantile beta to quantile 1-beta
    compare np.linalg.norm(p) in points
    """
    if beta<=0:
        print("beta<=0 means to keep all points")
        return points, weights
    if beta>=0.5:
        raise ValueError("beta>=0.5 means to drop out all points")
    
    if weights is None:
        upper = quantile(points, None, 1 - beta)
        lower = quantile(points, None, beta)
    else:
        upper = quantile(points, weights, 1 - beta)
        lower = quantile(points, weights, beta)
        
    if weights is None:
        points = [p for p in points 
                  if np.linalg.norm(lower) < np.linalg.norm(p) < np.linalg.norm(upper)]
        return points, None
    new_points=[]
    new_weights=[]
    for p, ws in zip(points, weights):
        if (np.linalg.norm(lower) < np.linalg.norm(p) < np.linalg.norm(upper)):
            new_points.append(p)
            new_weights.append(ws)
    return new_points, new_weights

def _gamma_mean_initializer(points, weights, initial):
    if initial == 'mean':
        return mean(points, weights)
    elif initial == 'median':
        return median(points, weights)

def gamma_mean_1D(points, weights=None, history_points=None, gamma = 0.1, max_iter=10, tol = 1e-7, remove=False, beta=0.1, initial = 'mean'):
    """
    We use element-wise mu & sigma,
    equal to diagnol Sigma,
    gamma_mean
    """
    if history_points is None:
        mu_hat = _gamma_mean_initializer(points, weights, initial)
        sigma_hat = std(points, weights)
    else:
        mu_hat = _gamma_mean_initializer(points, weights, initial)
        sigma_hat = std(history_points, weights)
    #sigma_hat = np.diag(np.cov(np.transpose(points)))
    
    """
    tol should consider the scale of points
    np.multiply(np.abs(mu_hat), tol) 
    """
    tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), tol), tol )

    if remove:
        """
        remove the extreme points
        """
        points, weights = ext_remove(points, weights, beta=beta)
    #the similar entries do not need to update
    index = (sigma_hat > tol)       
    if np.sum(index,axis=None)==0:
        return mu_hat
    
    for _ in range(max_iter):
        if history_points is None:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.divide(np.subtract(d[index], mu_hat[index]), sigma_hat[index]+1e-5), [-1])) ))
                for d in points]
        else:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.divide(np.subtract(d[index], mu_hat[index]), sigma_hat[index]), [-1])) ))
                for d in history_points]
        
        if np.all(np.array(d_gamma)==0):
            return mu_hat
        
        if weights is None:
            mu_hat[index] = mean(points, d_gamma )[index] #d_gamma serves as weight to mean
            sigma_hat[index] = np.sqrt((1+gamma)*mean(np.square(np.subtract(points,mu_hat)), 
                                               d_gamma ))[index]
        else:
            mu_hat[index] = mean(points, weights=np.multiply(d_gamma,weights) )[index]
            sigma_hat[index] = np.sqrt((1+gamma)*mean(np.square(np.subtract(points,mu_hat)), 
                                               np.multiply(d_gamma,weights) ))[index]
        index = (sigma_hat > tol)
        if np.sum(index,axis=None)==0:
            return mu_hat
        #if remove:
        #    #remove the extreme points
        #    points, weights = ext_remove(points, weights, beta=beta)
    return mu_hat #.astype(points.dtype)


def simple_gamma_mean(points, weights=None, history_points=None, gamma = 0.1, max_iter=10, tol = 1e-7, remove=False, beta=0.1, initial = 'mean'):
    """
    Do not consider cov inverse
    """
    if history_points is None:
        mu_hat = _gamma_mean_initializer(points, weights, initial)
        sigma_hat = std(points, weights)
    else:
        mu_hat = _gamma_mean_initializer(points, weights, initial)
        sigma_hat = std(history_points, weights)
    #sigma_hat = np.diag(np.cov(np.transpose(points)))
    
    """
    tol should consider the scale of points
    np.multiply(np.abs(mu_hat), tol) 
    """
    tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), tol), tol)

    if remove:
        """
        remove the extreme points
        """
        points, weights = ext_remove(points, weights, beta=beta)
    #the similar entries do not need to update
    index = (sigma_hat > tol)       
    if np.sum(index,axis=None)==0:
        return mu_hat
    
    for _ in range(max_iter):
        if history_points is None:
            #d_gamma list length: m
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.subtract(d[index], mu_hat[index]),[-1])) ))
                for d in points] #points shape: (m,p) 
        else:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.subtract(d[index], mu_hat[index]),[-1])) ))
                for d in history_points]
        if np.all(np.array(d_gamma)==0):
            return mu_hat
        
        if weights is None:
            mu_hat[index] = mean(points, d_gamma )[index]
        else:
            mu_hat[index] = mean(points, weights=np.multiply(d_gamma,weights) )[index]
    return mu_hat #.astype(points.dtype)

def dim_reduce(points, weights=None, method='pca', dim = None):
    """
    points: (m, p)
    cov--> (p, p)
    A v = lambda v => AV= Sigma V.transpose (V columns v1,v2,...)
    A = USV.transpose
    U: shape (p, p)
    S: eigenvalues 
    VT: shape (m, m)
    select max dim eigenvalues
    get transform_map by VT[:dim,:]    
    (p, dim)
    and inverse_transeform_map by transpose VT[:dim,:]
    (dim, p)
    
    return transform_map, inverse_transeform_map
    lowdim_data = np.dot(points, transform_map)
    approx_estimate = np.dot(lowdim_data, inverse_transeform_map)
    """
    points = [np.asarray(p).reshape([-1]) for p in points]
    if method=='pca':
        _, sigma, v = svd(cov(points, weights))
        expla_var = np.divide(np.cumsum(sigma),np.maximum(np.sum(sigma),1e-5))
        thred = 0.95
        if dim is None:
            dim = np.sum(expla_var<=thred)+1
        else:
            dim = np.minimum(np.sum(expla_var<=thred)+1,dim)
        return np.transpose(v[:dim,:]), np.array(v[:dim,:])
    elif method=='truncated_svd':
        return
    elif method=='kernal_pca':  
        return
    elif method=='sparse_pca':
        return
    elif method=='incremental_pca':
        return

def gamma_mean_2D(points, weights=None, history_points=None, gamma = 0.1, max_iter=10, tol = 1e-7, dim_red=False, red_method='pca', initial = 'mean'):
    """
    gamma_mean
    """
    original_tol=tol
    shape = np.shape(points[0])
    points = [np.asarray(p).reshape([-1]) for p in points]
    history_points = [np.asarray(p).reshape([-1]) for p in history_points]
    if history_points is None:
        mu_hat = _gamma_mean_initializer(points, weights, initial)
        sigma_hat = cov(points, weights)
    else:
        mu_hat = _gamma_mean_initializer(points, weights, initial)
        sigma_hat = cov(history_points, weights)
    
    """
    cov allow weights have some 0s
    """
    tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), tol), tol )

    #the similar entries do not need to update
    index = (sigma_hat > tol)       
    if np.sum(index,axis=None)==0:
        return mu_hat.reshape(shape)

    if dim_red:
        """
        dimension reduction
        compute in low dimension and go back to original dim after computation
        """
        transform_map, inverse_transeform_map = dim_reduce(points, weights, red_method)
        points = np.dot(points, transform_map)
        if weights is None:
            mu_hat = mean(points, None)
            sigma_hat = cov(points, None)
        else:
            mu_hat = mean(points, weights)
            sigma_hat = cov(points, weights)
        tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), original_tol), original_tol )
        #the similar entries do not need to update
        index = (sigma_hat > tol)  
    """
    @Todo, what if np.linalg.inv(sigma_hat) not exist?
    Face computation issue in inverse computing
    some columns=0, inverse does not exist 
    """
    for _ in range(max_iter):
        if history_points is None:
            d_gamma=[np.exp(-(gamma/2) * 
                np.dot(np.dot(np.subtract(d, mu_hat),
                    np.linalg.inv(sigma_hat)),
                    np.transpose(np.subtract(d, mu_hat))))
                for d in points]
        else:
            d_gamma=[np.exp(-(gamma/2) * 
                np.dot(np.dot(np.subtract(d, mu_hat),
                    np.linalg.inv(sigma_hat)),
                    np.transpose(np.subtract(d, mu_hat))))
                for d in history_points]
        if np.all(np.array(d_gamma)==0):
            if dim_red:
                """
                go back to original dim after computation
                """
                mu_hat = np.dot(mu_hat, inverse_transeform_map)
            return mu_hat.reshape(shape)
        
        if dim_red or weights is None:
            mu_hat = mean(points, d_gamma )
            sigma_hat = cov(points, d_gamma)
        else:
            mu_hat = mean(points, weights=np.multiply(d_gamma,weights) )
            sigma_hat = cov(points, np.multiply(d_gamma,weights) )
        index = (sigma_hat > tol)
        if np.sum(index,axis=None)==0:
            if dim_red:
                """
                go back to original dim after computation
                """
                mu_hat = np.dot(mu_hat, inverse_transeform_map)
            return mu_hat.reshape(shape)
    if dim_red:
        """
        go back to original dim after computation
        """
        mu_hat = np.dot(mu_hat, inverse_transeform_map)
    return mu_hat.reshape(shape)#.astype(points.dtype)


'''
gamma mean
'''
def gamma_mean(points, weights=None, history_points=None, compute = "1D", gamma = 0.1, max_iter=10, 
               tol = 1e-7, remove=False, beta=0.1, dim_red=False, red_method='pca', initial = 'mean'):
    if compute=="1D":
        return gamma_mean_1D(points=points, weights=weights, history_points=None, 
                             gamma = gamma, max_iter = max_iter, 
                             tol = tol, remove=remove, beta=beta, initial = initial)
    elif compute=="simple":
        return simple_gamma_mean(points=points, weights=weights, history_points=None, 
                                 gamma = gamma, max_iter = max_iter, 
                                 tol = tol, remove=remove, beta=beta, initial = initial)
    elif compute=="2D":
        return gamma_mean_2D(points=points, weights=weights, history_points=None, 
                             gamma = gamma, max_iter = max_iter, 
                             tol = tol, dim_red=dim_red, red_method=red_method, initial = initial)

'''
Geometric Median
'''
def geometric_median(points, weights=None, max_iter = 1000, tol = 1e-7):
    """
    Use Weiszfeld's method
    """
    mu = mean(points, weights)
    def distance_func(x):
        #return np.linalg.norm(np.subtract(points,x), axis=1)
        return np.array([np.linalg.norm(np.subtract(w.reshape([-1]),x.reshape([-1])), axis=0) for w in points])   
    distances = distance_func(mu)
    for _ in range(max_iter):
        prev_mu = mu
        if weights is None:
            beta_weights = 1 / np.maximum(1e-5, distances)
        else:
            beta_weights = weights / np.maximum(1e-5, distances)
        mu = mean(points,beta_weights) #weight average

        distances = distance_func(mu)
        mu_movement = np.sqrt((np.subtract(prev_mu, mu)**2).sum())

        if mu_movement <= tol:
            break
    return mu