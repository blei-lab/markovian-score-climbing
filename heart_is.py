import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.core import getval
from autograd.misc.optimizers import adam
from autograd.extend import notrace_primitive
from scipy.stats import norm as norm_extra

from heart_helper import *


# Initialize
seed_data = npr.RandomState(1234)
seed = npr.RandomState(1234)
S = 10
init_scale = 0.1

@notrace_primitive
def generate_samples(params, seed, verbose):
    mu, log_sigma = params
    
    # Propose values 
    z = mu[:,None] + np.exp(log_sigma[:,None])*seed.normal(size=(nz,S))

    # Compute weights
    logw = log_weights(params, z, train_data)
    maxLogW = np.max(logw)
    uw = np.exp(logw-maxLogW)
    unw = np.exp(logw)
    w = uw / np.sum(uw)

    return w, z

def objective(params, iter, verbose=False):
    w, z = generate_samples(params, seed, verbose)
    return -np.sum(log_posteriorapprox(params, z)*w)

objective_grad = grad(objective)

num_iters = 10000
num_indep = 100
step_size = 0.01
step_fun = lambda x: step_size/(x+1.)

pred_error = np.zeros(num_indep)
post_pred_val = []
opt_params = []

def print_perf(params, iter, grad):
    m, ls = params
    mu_vec[iter] = m
    logsigma_vec[iter] = ls
    if iter % 100 == 0:
        bound = np.mean(objective(params, iter, False))
        message = "{:15}|{:20}|".format(iter, bound)
        print(message)
    
for i in range(num_indep):
    print('Split: '+str(i))
    train_data, test_data, ny, nz = load_data(seed_data)
    
    init_mu = init_scale*seed.normal(size=nz)
    init_logsigma = 0.5+init_scale*seed.normal(size=nz)

    init_params = (init_mu, init_logsigma)
    zprev = init_scale*seed.normal(size=(nz, 2))

    mu_vec = np.zeros((num_iters,nz))
    logsigma_vec = np.zeros((num_iters,nz))
    
    print("     Epoch     |    Objective  ")
    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=num_iters, callback=print_perf)
    opt_params.append(optimized_params)
    # Prediction error
    test_outcome, test_regressors = test_data
    mu_opt = np.mean(mu_vec[-150:],axis=0)
    sigma_opt = np.diag(np.mean(np.exp(2.*logsigma_vec[-150:]),axis=0))
    xSigmax = np.sum(np.dot(test_regressors,sigma_opt)*test_regressors,axis=1)
    prob = norm.cdf(np.dot(test_regressors, mu_opt)/np.sqrt(1.+xSigmax))
    test_pred = (prob>0.5).astype('float')
    pred_error[i] = 1.-np.sum(test_pred == test_outcome)/float(len(test_outcome))
    post_pred_val.append(prob**(test_outcome) * (1.-prob)**(1.-test_outcome))

np.save('results/heart_S'+str(S)+'_stepsize'+str(step_size)+'_prederror_is.npy', pred_error)
np.save('results/heart_S'+str(S)+'_stepsize'+str(step_size)+'_params_is.npy', np.array(opt_params))
np.save('results/heart_S'+str(S)+'_stepsize'+str(step_size)+'_postpredval_is.npy', np.array(post_pred_val))

print(np.array(opt_params).shape)
print(np.array(post_pred_val))
print(pred_error)
print(np.mean(pred_error))