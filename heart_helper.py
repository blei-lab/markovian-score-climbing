import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mult_norm

from autograd import grad
from autograd.extend import notrace_primitive
from scipy.stats import norm as norm_extra

# Sample from discrete distribution
def discretesampling(w, seed):
    u = seed.rand()
    bins = np.cumsum(w)
    return np.digitize(u,bins)

# Load data
def load_data(seed=npr.RandomState(1234),complete=False):
    fname = 'data/processed.cleveland.data'
    complete_data = np.genfromtxt(fname, delimiter=',')
    complete_data = complete_data[~np.isnan(complete_data).any(axis=1)]
    regressor_data = complete_data[:,:-1]
    regressor_data = np.insert(regressor_data, 0, 1, axis=1)
    outcome_data = complete_data[:,-1]
    # Binarize
    outcome_data = (outcome_data > 0).astype('float').reshape((len(outcome_data),1))

    ny = regressor_data.shape[0]
    nz = regressor_data.shape[1]
    
    if complete:
        num_train = 297
    else:
        num_train = 267
    num_test = ny-num_train
    idx_test = seed.permutation(ny)
    train_data = (outcome_data[idx_test[:num_train]], regressor_data[idx_test[:num_train]])
    test_data = (outcome_data[idx_test[num_train:]].reshape(num_test), regressor_data[idx_test[num_train:]])
    
    return train_data, test_data, ny, nz

def log_likelihood(y_data, lcdf, lsf):
    # y_data is array (ny, 1), lcdf and lsf are array (ny, S)
    return np.sum(y_data*lcdf + (1.-y_data)*lsf, axis=0)

def log_prior(latent):
    # latent is array (nz, S)
    return np.sum(-0.5*latent**2 - 0.5*np.log(2.*np.pi), axis=0)

def log_posteriorapprox(params, latent):
    mu, log_sigma = params
    return np.sum(-0.5*(latent-mu[:,None])**2/np.exp(2.*log_sigma[:,None]) - 0.5*np.log(2.*np.pi) - log_sigma[:,None], axis=0)

def compute_covariance(params):
    mu, log_sigma, lt_vec = params
    n = mu.shape[0]
    lt_mat = np.zeros((n,n))
    count = 0
    for i in range(n-1):
        lt_mat = lt_mat + np.diag(lt_vec[count:count+i+1], k=-n+i+1)
        count += i+1
    Sigma_chol = np.diag(np.exp(log_sigma)) + lt_mat
    Sigma = np.dot(Sigma_chol, Sigma_chol.T)
    return Sigma

def log_posteriorapprox_mult(params, latent):
    # Latent is num_samples x dim_latent
    mu, log_sigma, lt_vec = params
    Sigma = compute_covariance(params)
    return mult_norm.logpdf(latent, mu, Sigma)

def log_weights_mult(params, latent, data):
    # params is (mu, log_sigma), latent is array (nz, S), x_data is array (ny, nz), y_data is array (ny, 1)
    y_data, x_data = data
    lcdf = norm.logcdf(np.dot(x_data, latent.T))
    lsf = norm_extra.logsf(np.dot(x_data, latent.T))
    return log_prior(latent.T) + log_likelihood(y_data, lcdf, lsf) - log_posteriorapprox_mult(params, latent)

def log_weights(params, latent, data):
    # params is (mu, log_sigma), latent is array (nz, S), x_data is array (ny, nz), y_data is array (ny, 1)
    y_data, x_data = data
    lcdf = norm.logcdf(np.dot(x_data, latent))
    lsf = norm_extra.logsf(np.dot(x_data, latent))
    return log_prior(latent) + log_likelihood(y_data, lcdf, lsf) - log_posteriorapprox(params, latent)

def log_likelihood_ep(latent, data):
    y_data, x_data = data
    lcdf = norm.logcdf(np.sum(x_data*latent))
    lsf = norm.logcdf(-np.sum(x_data*latent))
    return y_data*lcdf + (1.-y_data)*lsf

def log_weights_ep(latent, data):
    # latent is array (nz, S), x_data is array (ny, nz), y_data is array (ny, 1)
    y_data, x_data = data
    lcdf = norm.logcdf(np.dot(x_data, latent))
    lsf = norm_extra.logsf(np.dot(x_data, latent))
    return log_likelihood(y_data, lcdf, lsf)