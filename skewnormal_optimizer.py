"""Some standard gradient-based stochastic optimizers.
These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.
These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""
from __future__ import absolute_import
from builtins import range

import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f


def unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g))
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, _callback, *args, **kwargs))

    return _optimize

@unflatten_optimizer
def sgd(grad, x, callback=None, num_iters=200, step_size=0.1):
    """Vanilla stochastic gradient descent.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        x = x - step_size * g / float(i+1)
    return x

logsf = primitive(scipy.stats.norm.logsf)
logpdf = primitive(scipy.stats.norm.logpdf)

defvjp(logsf, 
       lambda ans, x, loc=0.0, scale=1.0: 
       unbroadcast_f(x, lambda g: -g * np.exp(logpdf(x, loc, scale) - logsf(x, loc, scale))),
       lambda ans, x, loc=0.0, scale=1.0: 
       unbroadcast_f(loc, lambda g: g * np.exp(logpdf(x, loc, scale) - logsf(x, loc, scale))),
       lambda ans, x, loc=0.0, scale=1.0: 
       unbroadcast_f(scale, lambda g: g * np.exp(logpdf(x, loc, scale) - logsf(x, loc, scale)) * (x - loc) / scale))