import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.extend import notrace_primitive
from scipy.stats import norm as norm_extra

# Sample from discrete distribution
def discretesampling(w, seed):
    u = seed.rand()
    bins = np.cumsum(w)
    return np.digitize(u,bins)

# Resampling
def resampling(w, seed):
    # Systematic resampling
    N = w.shape[0]
    bins = np.cumsum(w)
    u = (seed.rand()+np.arange(N))/N
    return np.digitize(u, bins)

# Load data
def load_data(number):
    fname = 'data/FRB_H10.csv'
    observations = np.loadtxt(fname, delimiter=",", usecols=range(5,23), skiprows=6)
    observation = np.log(observations[1:,number])-np.log(observations[:-1,number])
    print(observations.shape)
    return observation

# Log-likelihood N(0, exp(log_beta+log_vol))
def log_likelihood(log_volatility, y_data, mod_params):
    mu, logphi, log_sigma, log_beta = mod_params
    phi = np.tanh(logphi)
    sigma2 = np.exp(log_volatility+log_beta)
    return -0.5*y_data**2/sigma2 - 0.5*np.log(2.*np.pi) - 0.5*log_volatility - 0.5*log_beta

# Log-prior N(mu, sigma2/(1-phi2))
def log_prior(log_volatility, mod_params):
    mu, logphi, log_sigma, log_beta = mod_params
    phi = np.tanh(logphi)
    sigma2 = np.exp(log_sigma*2.)/(1.-phi**2)
    return -0.5*(log_volatility-mu)**2/sigma2 - 0.5*np.log(2.*np.pi) - 0.5*np.log(sigma2)

# Log-transition N(mu+phi(log_vol-mu), sigma2)
def log_transition(log_volatility, log_volatility_prev, mod_params):
    mu, logphi, log_sigma, log_beta = mod_params
    phi = np.tanh(logphi)
    mean = mu + phi*(log_volatility_prev-mu)
    sigma2 = np.exp(log_sigma*2.)
    return -0.5*(log_volatility-mean)**2/sigma2 - 0.5*np.log(2.*np.pi) - 0.5*np.log(sigma2)

# eval mod_params
@notrace_primitive
def eval_modparams(mod_params):
    mu, logphi, log_sigma, log_beta = mod_params
    phi = np.tanh(logphi)
    return mu, phi, log_sigma, log_beta

# Log-approx conditionals
def log_variational_approx(var_params, mod_params, t, log_volatility, log_volatility_prev):
    mu, phi, log_sigma, log_beta = eval_modparams(mod_params)
    infvec, log_precision = var_params
    infval = infvec[t]
    log_precval = log_precision[t]
    var2 = np.exp(-log_precval)
    mean2 = var2*infval
    if t > 0:
        mean1 = mu + phi*(log_volatility_prev-mu)
        var1 = np.exp(log_sigma*2.)
    else:
        mean1 = mu
        var1 = np.exp(log_sigma*2.)/(1.-phi**2)
    var = 1./(1./var2+1./var1)
    mean = var*(mean1/var1 + mean2/var2)
    
    return -0.5*(log_volatility-mean)**2/var - 0.5*np.log(2.*np.pi) - 0.5*np.log(var)

# Simulate variational approx
def sim_variational_approx(t, log_volatility_prev, var_params, mod_params, seed):
    mu, phi, log_sigma, log_beta = eval_modparams(mod_params)
    infvec, log_precision = var_params
    infval = infvec[t]
    log_precval = log_precision[t]
    var2 = np.exp(-log_precval)
    mean2 = var2*infval
    if t > 0:
        mean1 = mu + phi*(log_volatility_prev-mu)
        var1 = np.exp(log_sigma*2.)
    else:
        mean1 = mu
        var1 = np.exp(log_sigma*2.)/(1.-phi**2)
    var = 1./(1./var2+1./var1)
    mean = var*(mean1/var1 + mean2/var2)
    
    return mean + np.sqrt(var)*seed.normal(size=len(log_volatility_prev))

# Log-approx joint
def log_approx(var_params, log_volatility, mod_params):
    T = log_volatility.shape[0]-1
    output = log_variational_approx(var_params, mod_params, 0, log_volatility[0], log_volatility[-1])
    for t in range(1,T+1):
        output = output + log_variational_approx(var_params, mod_params, t, log_volatility[t], log_volatility[t-1])
    return output

def log_target(t, log_volatility, log_volatility_prev, y_data, mod_params):
    output = np.zeros_like(log_volatility)
    if t > 0:
        output = output + log_transition(log_volatility, log_volatility_prev, mod_params)
        output = output + log_likelihood(log_volatility, y_data[t-1], mod_params)
    else:
        output = output + log_prior(log_volatility, mod_params)
    return output

# Log-target joint
def log_target_complete(log_volatility, y_data, mod_params):
    T = log_volatility.shape[0]-1
    output = log_prior(log_volatility[0], mod_params)
    for t in range(1,T+1):
        output = output + log_transition(log_volatility[t], log_volatility[t-1], mod_params)
        output = output + log_likelihood(log_volatility[t], y_data[t-1], mod_params)
    return output

def log_weights(t, log_volatility, log_volatility_prev, y_data, var_params, mod_params):
    return log_target(t, log_volatility, log_volatility_prev, y_data, mod_params) - log_variational_approx(var_params, mod_params, t, log_volatility, log_volatility_prev)

# Log-twisting
def log_twisting(t, log_volatility, var_params):
    infvec, log_precision = var_params
    infval = infvec[t]
    log_precval = log_precision[t]
    return -0.5*np.exp(log_precval)*log_volatility**2 + infval*log_volatility

def log_target_twisted(t, log_volatility, log_volatility_prev, y_data, var_params, mod_params):
    output = np.zeros_like(log_volatility)
    
    # Twisting potential not used at t=T and only likelihood
    T = len(y_data)
    if t < T:
        output = output + log_twisting(t, log_volatility, var_params)
    else:
        output = output + log_likelihood(log_volatility, y_data[t-1], mod_params)
    
    # Transition/prior at each step, remove previous twisting 
    if t > 0:
        output = output + log_transition(log_volatility, log_volatility_prev, mod_params)
        output = output - log_twisting(t-1, log_volatility_prev, var_params)
    else:
        output = output + log_prior(log_volatility, mod_params)
    
    # Previous observation
    if t > 1:
        output = output + log_likelihood(log_volatility_prev, y_data[t-2], mod_params)
    return output

def log_weights_twisting(t, log_volatility, log_volatility_prev, y_data, var_params, mod_params):
    return log_target_twisted(t, log_volatility, log_volatility_prev, y_data, var_params, mod_params) - log_variational_approx(var_params, mod_params, t, log_volatility, log_volatility_prev)

# BOOTSTRAP
def sim_bootstrap(t, log_volatility_prev, mod_params, seed):
    mu, logphi, sigma, log_beta = mod_params
    phi = np.tanh(logphi)
    if t > 0:
        mean = mu + phi*(log_volatility_prev-mu)
        sigma2 = sigma**2
    else:
        mean = mu
        sigma2 = sigma**2/(1.-phi**2)
        
    return mean + np.sqrt(sigma2)*seed.normal(size=len(log_volatility_prev))

def log_weights_bootstrap(t, log_volatility, log_volatility_prev, y_data, mod_params):
    output = np.zeros_like(log_volatility)
    if t > 0:
        output = log_likelihood(log_volatility, y_data[t-1], mod_params)
    return output