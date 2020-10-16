import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.core import getval
from skewnormal_optimizer import sgd
from autograd.extend import notrace_primitive

# Define true normal parameters
loc = 0.5
scale = 2.
shape = 5.
# Compute true expectations
delta = shape/np.sqrt(1.+shape**2)
mu_true = loc+scale*delta*np.sqrt(2./np.pi)
sigma2_true = (1.-2*delta**2/np.pi)*scale**2

def logq(params, x):
    mu, log_sigma = params
    std_val = (x-mu)/np.exp(log_sigma)
    return -0.5*std_val**2 - 0.5*np.log(2.*np.pi) - log_sigma

def logp(x):
    std_val = (x-loc)/scale
    return np.log(2.)-np.log(scale)+norm.logpdf(std_val)+norm.logcdf(shape*std_val)

S = 2
@notrace_primitive
def generate_samples(params, seed):
    mu, log_sigma = params
    x = mu + np.exp(log_sigma)*seed.normal(size=S)
    logw = logp(x) - logq(params, x)
    maxLogW = np.max(logw)
    uw = np.exp(logw-maxLogW)
    w = uw / np.sum(uw)
    return w, x

seed = npr.RandomState(0)
def objective(params, iter):
    w, x = generate_samples(params, seed)
    return -np.sum(w*logq(params, x))

objective_grad = grad(objective)

init_params = (mu_true, 0.5*np.log(sigma2_true))
num_iters = 1000000
step_size = 0.5

params_vec = np.zeros((num_iters,2))
   
print("     Epoch     |    Objective  ")
def print_perf(params, iter, grad):
    m, ls = params
    params_vec[iter, 0] = m
    params_vec[iter, 1] = ls
    if iter % 5000 == 0:
        bound = np.mean(objective(params, iter))
        message = "{:15}|{:20}|".format(iter, bound)
        print(message)

# The optimizers provided can optimize lists, tuples, or dicts of parameters.
optimized_params = sgd(objective_grad, init_params, step_size=step_size,
                        num_iters=num_iters, callback=print_perf)

np.save('results/skewnormal_S'+str(S)+'_is.npy', params_vec)