import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.core import getval
from autograd.misc.optimizers import adam
from autograd.extend import notrace_primitive
from scipy.stats import norm as norm_extra

from stochasticvol_helper import *


# Initialize
seed = npr.RandomState(1234)
init_scale = 0.1
N = 10

num_iters = 2000
num_indep = 18
step_size = 0.1
step_fun = lambda x: step_size/(x+1.)

logZ = np.zeros(num_indep)

def compute_likelihood(params, seed, Np):
    var_params, mod_params = params
    latent_samples = np.zeros((Np,T+1))
    logW = np.zeros(Np)
    W = np.exp(logW)
    W /= np.sum(W)
    logZ = 0.
    
    for t in range(T+1):
        # Regular resampling
        if t > 0:
            ancestors = resampling(W, seed)
            latent_samples[:,:t] = latent_samples[ancestors,:t]
        
        # Propagate
        latent_samples[:,t] = sim_variational_approx(t, latent_samples[:,t-1], var_params, mod_params, seed)
            
        # Weights
        logW = log_weights(t, latent_samples[:,t], latent_samples[:,t-1], observations, var_params, mod_params)
        maxLogW = np.max(logW)
        W = np.exp(logW-maxLogW)
        logZ += maxLogW + np.log(np.sum(W)) - np.log(Np)
        W /= np.sum(W)
        
    return logZ

@notrace_primitive
def generate_samples(params, seed):
    var_params, mod_params = params
    latent_samples = np.zeros((N,T+1))
    logW = np.zeros(N)
    W = np.exp(logW)
    W /= np.sum(W)
    
    for t in range(T+1):
        # Regular resampling
        if t > 0:
            ancestors = resampling(W, seed)
            latent_samples[:,:t] = latent_samples[ancestors,:t]
        
        # Propagate
        latent_samples[:,t] = sim_variational_approx(t, latent_samples[:,t-1], var_params, mod_params, seed)
            
        # Weights
        logW = log_weights(t, latent_samples[:,t], latent_samples[:,t-1], observations, var_params, mod_params)
        maxLogW = np.max(logW)
        W = np.exp(logW-maxLogW)
        W /= np.sum(W)
        
    B = discretesampling(W, seed)
    return latent_samples[B]

def objective(params, iter):
    var_params, mod_params = params
    latent_prev = generate_samples(params, seed)
    return -log_target_complete(latent_prev, observations, mod_params)-log_approx(var_params, latent_prev, mod_params)

objective_grad = grad(objective)


def print_perf(params, iter, grad):
    var_params, mod_params = params
    m, ls = var_params
    inf_vec[iter] = m
    loglbda_vec[iter] = ls
    
    if iter % 100 == 0:
        bound = compute_likelihood(params, seed, Np=N)
        message = "{:15}|{:20}|".format(iter, bound)
        print(message)
    
for i in range(num_indep):
    print('Split: '+str(i))
    fname = 'data/'
    observations = load_data(i)
    T = observations.shape[0]
    nz = T+1
    
    init_mu = init_scale*seed.normal()
    init_phi = init_scale*seed.normal()
    init_logsigma = init_scale*seed.normal()
    init_beta = init_scale*seed.normal()
    init_modparams = (init_mu, init_phi, init_logsigma, init_beta)
    
    init_infvec = init_scale*seed.normal(size=nz)
    init_loglbda = 0.5+init_scale*seed.normal(size=nz)
    init_varparams = (init_infvec, init_loglbda)

    init_params = (init_varparams, init_modparams)
    latent_prev = init_scale*seed.normal(size=nz)
    
    inf_vec = np.zeros((num_iters,nz))
    loglbda_vec = np.zeros((num_iters,nz))
    logZ_tmp = np.zeros(num_iters)
    
    print("     Epoch     |    Objective  ")
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=num_iters, callback=print_perf)
    
    logZ[i] = compute_likelihood(optimized_params, seed, Np=10000)
    print(logZ[i])

np.save('results/stochasticvol_N'+str(N)+'_stepsize'+str(step_size)+'_infvec_smc.npy', inf_vec)
np.save('results/stochasticvol_N'+str(N)+'_stepsize'+str(step_size)+'_loglbda_smc.npy', loglbda_vec)
np.save('results/stochasticvol_N'+str(N)+'_stepsize'+str(step_size)+'_logZ_smc.npy', logZ)