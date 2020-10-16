import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad, hessian
from autograd.core import getval
from autograd.extend import notrace_primitive
from scipy.stats import norm as norm_extra

from heart_helper import *


# Initialize
seed = npr.RandomState(1234)
num_iters = 2
num_indep = 100
pred_error = np.zeros(num_indep)
post_pred_val = []

for i in range(num_indep):
    print('Split: '+str(i))
    train_data, test_data, ny, nz = load_data(seed)
    train_outcome, train_regressor = train_data
    ny = train_outcome.shape[0]
    
    # Init EP
    Lambda = np.zeros((ny, nz, nz))
    nu = np.zeros((ny, nz))
    
    # Number of iterations of the EP algorithm
    for j in range(num_iters):
        #print('Iter: '+str(j))
        # One loop of all the approximate factors
        for k in range(int(ny)):
            # Set up IS
            y, x = train_outcome[k], train_regressor[k]
            nu_tmp = np.sum(nu,axis=0)-nu[k]
            lambda_tmp = np.eye(nz) + np.sum(Lambda, axis=0) - Lambda[k]
            sigma_tmp = np.linalg.inv(lambda_tmp)
            mu_tmp = np.dot(sigma_tmp, nu_tmp)
            
            # Update
            qp_form = np.inner(x,np.dot(sigma_tmp,x))
            z = np.inner(x,mu_tmp)/np.sqrt(1.+qp_form) 
            if y > 0:
                grad_mu = x*norm.pdf(z)/norm.cdf(z)/np.sqrt(1.+qp_form)
                grad_sig = -0.5*np.outer(x,x)*z*norm.pdf(z)/norm.cdf(z)/(1.+qp_form)
                grad_cov = np.outer(grad_mu,grad_mu)-2.*grad_sig
            else:
                z = -z
                grad_mu = -x*norm.pdf(z)/norm.cdf(z)/np.sqrt(1.+qp_form)
                grad_sig = -0.5*np.outer(x,x)*z*norm.pdf(z)/norm.cdf(z)/(1.+qp_form)
                grad_cov = np.outer(grad_mu,grad_mu)-2.*grad_sig
                
            mu_hat = mu_tmp + np.dot(sigma_tmp, grad_mu)
            sigma_hat = sigma_tmp - np.dot(sigma_tmp, np.dot(grad_cov,sigma_tmp))
            lambda_hat = np.linalg.inv(sigma_hat)
            
            # Update factors
            Lambda[k] = lambda_hat - lambda_tmp
            nu[k] = np.dot(lambda_hat, mu_hat) - nu_tmp
            
    # Compute final approximation
    lambda_mat = np.eye(nz) + np.sum(Lambda, axis=0)
    nu_vec = np.sum(nu, axis=0)
    sigma_opt = np.linalg.inv(lambda_mat)
    mu_opt = np.dot(sigma_opt, nu_vec)
    # Prediction error
    test_outcome, test_regressors = test_data
    xSigmax = np.sum(np.dot(test_regressors,sigma_opt)*test_regressors,axis=1)
    prob = norm.cdf(np.dot(test_regressors, mu_opt)/np.sqrt(1.+xSigmax))
    test_pred = (prob>0.5).astype('float')
    pred_error[i] = 1.-np.sum(test_pred == test_outcome)/float(len(test_outcome))
    post_pred_val.append(prob**(test_outcome) * (1.-prob)**(1.-test_outcome))


np.save('results/heart_numiter'+str(num_iters)+'_prederror_ep.npy', pred_error)
np.save('results/heart_numiter'+str(num_iters)+'_nu_ep.npy', nu_vec)
np.save('results/heart_numiter'+str(num_iters)+'_lambda_ep.npy', lambda_mat)
np.save('results/heart_numiter'+str(num_iters)+'_postpredval_ep.npy', np.array(post_pred_val))


#print(np.array(post_pred_val))
print(pred_error)
print(np.mean(pred_error))