import tensorflow as tf
import numpy as np

def array2tensor(array):
    
    return tf.expand_dims(tf.convert_to_tensor(array), 0)

def de_normalize_tensor(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_time = tf.concat([tf.zeros([state.shape[0], state.shape[1]-1]), tf.reshape((state[:,-1]+1)*state_norm_arr[-1]/2,[state.shape[0],1])],1)
    state_no_time = state * state_norm_arr
    mask = tf.concat([tf.ones([state.shape[0], state.shape[1]-1]), tf.zeros([state.shape[0], 1])],1)
    state_not_norm = state_no_time * mask + state_time * (1 - mask)

    return state_not_norm

def normalize_tensorOLD(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_norm_time = tf.concat([tf.zeros([state.shape[0], state.shape[1]-1]), tf.reshape(((state[:,-1]) / state_norm_arr[-1])*2 - 1,[state.shape[0],1])],1)
    state_norm_no_time = state / state_norm_arr
    mask = tf.concat([tf.ones([state.shape[0], state.shape[1]-1]), tf.zeros([state.shape[0], 1])],1)
    state_norm = state_norm_no_time * mask + state_norm_time * (1 - mask)

    return state_norm

def custom_logarithm(input):
    # Calculate the logarithms based on the non-zero condition
    sign_log = tf.math.sign(input)

    return sign_log*tf.math.log(tf.math.maximum(sign_log*input, 1e-7) + 1) 

def normalize_tensor(state, state_norm_arr):
    ''' Normalize the state tensor based on normalization factors '''
    # Normalize time component separately

    return tf.concat((state[:, :-1] / state_norm_arr[:-1], ((state[:, -1:] / state_norm_arr[-1]) * 2) - 1), axis=-1)

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


