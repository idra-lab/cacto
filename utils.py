import numpy as np

def de_normalize(state, state_norm_arr):
    ''' Retrieve state from normalized state '''
    state_not_norm  = np.empty_like(state)
    state_not_norm[:-1] = state[:-1] * state_norm_arr[:-1]
    state_not_norm[-1] = (state[-1] + 1) * state_norm_arr[-1]/2

    return state_not_norm

def normalize(state, state_norm_arr):
    ''' Normalize state '''
    state_norm  = np.empty_like(state)
    state_norm = state / state_norm_arr
    state_norm[-1] = state_norm[-1] * 2 -1

    return state_norm

import jax
import jax.numpy as jnp
import conf_double_integrator as conf
#import conf_manipulator as conf

def nn_eval(model, params, input):
    """Compute the output of a NN given an input in JAX."""
    
    if conf.NORMALIZE_INPUTS:
        input = input / conf.state_norm_arr

    if (conf.system_id in {'car', 'car_park'}) and conf.remap_angle:
        input = jnp.hstack([
            input[:, 0], input[:, 1], jnp.cos(input[:, 2]), jnp.sin(input[:, 2]),
            input[:, 3], input[:, 4], input[:, 5]
        ])
    elif conf.system_id == 'manipulator' and conf.remap_angle:
        input = jnp.hstack([
            jnp.cos(input[0]), jnp.sin(input[0]), jnp.cos(input[1]), jnp.sin(input[1]),
            jnp.cos(input[2]), jnp.sin(input[2]), input[3], input[4], input[5], input[6]
        ])
    elif conf.system_id == 'bicopter' and conf.remap_angle:
        input = jnp.hstack([
            input[:, 0], input[:, 1], jnp.cos(input[:, 2]), jnp.sin(input[:, 2]),
            input[:, 3], input[:, 4], input[:, 5], input[:, 6]
        ])

    return model.apply(params, input)

