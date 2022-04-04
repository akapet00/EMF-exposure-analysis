import itertools
import os
import time

import jax
import jax.numpy as jnp
import jax.nn as nn
from jax import vmap, jit, grad
from jax.example_libraries import optimizers
import numpy as np
import numpy.random as npr
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dosipy.utils.dataloader import load_antenna_el_properties
from utils import *

# load pre-computed source data
f = 26e9  # operating frequency of the antenna
antenna_data = load_antenna_el_properties(f)
Is = antenna_data.ireal.to_numpy() + antenna_data.iimag.to_numpy() * 1j
Is = np.asarray(Is)
xs = antenna_data.x.to_numpy()

rng = jax.random.PRNGKey(0)


def init_network_params(sizes, key):
    """Initialize network parameters."""
    keys = jax.random.split(key, len(sizes))
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = jax.random.split(key)
        return (scale * jax.random.normal(w_key, (n, m)),
                scale * jax.random.normal(b_key, (n, )))
    return [random_layer_params(m, n, key)
            for m, n, key in zip(sizes[:-1], sizes[1:], keys)]


def forward(params, X, scaler):
    """Forward pass."""
    output = X
    for w, b in params[:-1]:
        output = nn.tanh(w @ output + b)
    w, b = params[-1]
    output = w @ output + b
    return output * scaler


# vectorized mapping of network input, `X`, on `forward` function
batch_forward = vmap(forward, in_axes=(None, 0, None))


@jit
def loss_fn(params, batch, scaler):
    """Summed square error loss function."""
    X, y = batch
    y_pred = batch_forward(params, X, scaler)
    return jnp.sum(jnp.square(y_pred - y))


# derivative of the loss function
grad_fn = jit(grad(loss_fn))


@jit
def update(step, optim_state, batch, scaler):
    """Return current optimal state of the network."""
    params = optim_params(optim_state)
    grads = grad_fn(params, batch, scaler)
    optim_state = optim_update(step, grads, optim_state)
    return optim_state


# generate training data
Is_fn = interp1d(xs, np.abs(Is), kind='quadratic')
xs_interp = np.linspace(xs.min(), xs.max(), 961)
Is_interp = Is_fn(xs_interp)
xs_norm = normalize(xs_interp, xs_interp)
xs_data = jnp.array(xs_norm).reshape(-1, 1)
# the following scaling is a bit weird but is needed in order to keep
# gradients of the nn output wrt input (not parameters, but actual input) in
# the same scale with the ones that are computed via FDM
Is_norm = normalize(xs_interp, Is_interp)
Is_data = jnp.array(Is_norm).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(xs_data, Is_data,
                                                    test_size=0.25)

# set network hyperparameter and train
# to set the output of the nn in scale with the target data, we define scaler
scaler = np.abs(y_train).max()
step_size = 1e-3
n_epochs = 10_000
printout = int(n_epochs / 100.)
epochs = np.arange(0, n_epochs+1, step=printout)
batch_size = 128
momentum_mass = 0.9  # for momentum and adagrad
sizes = [1, 128, 256, 128, 1]

num_train = X_train.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)


def data_stream(num_train, num_batches):
    """Training data random generator."""
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield X_train[batch_idx], y_train[batch_idx]
            

batches = data_stream(num_train, num_batches)

optim_init, optim_update, optim_params = optimizers.adam(step_size)
init_params = init_network_params(sizes, rng)
optim_state = optim_init(init_params)
itercount = itertools.count()
 
loss_train, loss_test = [], []
params_list = []
start_time = time.time()
pbar = tqdm(range(n_epochs))
for epoch in pbar:
    start_epoch_time = time.time()
    for _ in range(num_batches):
        optim_state = update(next(itercount), optim_state, next(batches), scaler)
    epoch_duration = time.time() - start_epoch_time
    
    params = optim_params(optim_state)
    if (epoch == 0) or (epoch % printout == (printout - 1)):
        params_list.append(params)
        curr_loss_train_val = loss_fn(params, (X_train, y_train), scaler)
        curr_loss_test_val = loss_fn(params, (X_test, y_test), scaler)
        loss_train.append(curr_loss_train_val)
        loss_test.append(curr_loss_test_val)
        pbar.set_description(f'Loss (test): {curr_loss_test_val:.4e}')
training_duration = time.time() - start_time
print(f'Training time: {training_duration:.2f} s')

# choose params with the best performance on test set
best_params_idx = loss_test.index(min(loss_test))
params = params_list[best_params_idx]

# current distribution
Is_fit = batch_forward(params, xs_data.reshape(-1, 1), scaler)
Is_fit_inv_norm = inv_normalize(Is_fit, xs_interp)


def Is_nn(xs):
    """Current value at specific location, `xs`.
    
    Note: This is single-value wrapper for the forward pass function.
    """
    return forward(params, xs, scaler)[0]


# derivative of the current approximation function
grad_Is_nn = jit(vmap(grad(Is_nn)))


# save the data
np.save(os.path.join('data', f'x_at{int(f / 1e9)}GHz'),
        np.asarray(xs_interp))
np.save(os.path.join('data', f'current_at{int(f / 1e9)}GHz'),
        np.asarray(Is_fit_inv_norm))
np.save(os.path.join('data', f'grad_current_at{int(f / 1e9)}GHz'),
        np.asarray(grad_Is_nn(xs_data)))
