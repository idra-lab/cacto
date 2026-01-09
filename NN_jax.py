import jax
import flax
import optax
import pickle
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from typing import Callable, Any
from jax.nn.initializers import glorot_uniform, zeros
from flax.training.train_state import TrainState

class TrainStateC(TrainState):
    """Critic TrainState with an additional `target_params` field 
    for target network updates (used for soft critic update)."""
    target_params: flax.core.FrozenDict

class TrainStateA(TrainState):
    """Actor TrainState with an additional `prev_params` field 
    for storing the previous policy parameters (used in biased-sampling)."""
    prev_params: flax.core.FrozenDict

# Define Siren network implementation
class Sine(nn.Module):
    """Sine activation used in SIREN networks."""
    w0: float = 1.0
    
    @nn.compact
    def __call__(self, x):
        return jnp.sin(self.w0 * x)

def siren_first_layer_init(scale: float = 1.0, dtype: Any = jnp.float_):
    """Initialization for the first SIREN layer."""
    def init_fn(key, shape, dtype=dtype):
        fan_in = shape[0] if len(shape) == 2 else jnp.prod(shape[:-1])
        limit = scale / max(1.0, fan_in)
        return random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return init_fn

def siren_init(w0: float = 1.0, c: float = 6.0, dtype: Any = jnp.float_):
    """Initialization for subsequent SIREN layers."""
    def init_fn(key, shape, dtype=dtype):
        fan_in = shape[0] if len(shape) == 2 else jnp.prod(shape[:-1])
        limit = jnp.sqrt(c) / w0
        return random.uniform(key, shape, dtype, minval=-limit/fan_in, maxval=limit/fan_in)
    return init_fn

def he_uniform_bias(n):
    """He-uniform bias initializer."""
    def init(key, shape, dtype=jnp.float32):
        limit = jnp.sqrt(6.0 / n)
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return init

class SinusoidalRepresentationDense(nn.Module):
    """Dense layer with sinusoidal activation (SIREN block)."""
    features: int
    w0: float = 1.0
    c: float = 6.0
    use_bias: bool = True
    kernel_init: Callable = None
    first_layer: bool = False
    
    @nn.compact
    def __call__(self, inputs):
        # Choose the correct initialization based on layer type
        if self.kernel_init is None:
            if self.first_layer:
                kernel_init = siren_first_layer_init(self.c)
            else:
                kernel_init = siren_init(self.w0, self.c)
        else:
            kernel_init = self.kernel_init
            
        x = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=he_uniform_bias(n=self.features)
        )(inputs)
        
        return Sine(w0=self.w0)(x)

class ActorMLP(nn.Module):
    """
    Actor network (continuous actions).
    
    Attributes:
        conf (object): Configuration object containing the following attributes:
            NORMALIZE_INPUTS (bool): Whether to normalize the inputs.
            shift (float array): Shift values applied during normalization to center the input range in 0.
            state_norm_arr (float array): Normalization factors for the state inputs.
            remapped_indices (list): List of indices for remapping input angles.
            cos_element (list): List of indices of elements to be encoded as cosines.
            sin_element (list): List of indices of elements to be encoded as sines.
            features_actor (list): List of layer sizes for the actor MLP.
            u_max (float array): Maximum magnitude of the action output.
    """

    conf: Any

    @nn.compact
    def __call__(self, x):
        # Normalize input
        if self.conf.NORMALIZE_INPUTS:
            x = (x + self.conf.shift) / self.conf.state_norm_arr

        # Optional angle remapping (to [cos, sin])
        if self.conf.remapped_indices is not None:
            if x.ndim > 1:
                x = x[:,self.conf.remapped_indices]
                x = x.at[:,self.conf.cos_element].set(jnp.cos(x[:,self.conf.cos_element]))
                x = x.at[:,self.conf.sin_element].set(jnp.sin(x[:,self.conf.sin_element]))
            else:
                x = x[self.conf.remapped_indices]
                x = x.at[self.conf.cos_element].set(jnp.cos(x[self.conf.cos_element]))
                x = x.at[self.conf.sin_element].set(jnp.sin(x[self.conf.sin_element]))

        # Feedforward layers
        for feat in self.conf.features_actor[:-1]:
            x = nn.Dense(feat, kernel_init=glorot_uniform(), bias_init=zeros)(x)
            x = nn.leaky_relu(x, negative_slope=0.3)
        x = nn.Dense(self.conf.features_actor[-1], kernel_init=glorot_uniform(), bias_init=zeros)(x)

        # Output layer (tanh scaled by u_max)
        x = nn.tanh(x)*self.conf.u_max

        return x



class CriticSineMLP(nn.Module):
    """
    Critic network with sinusoidal activations (SIREN-style).
    
    Attributes:
        conf (object): Configuration object containing the following attributes:
            NORMALIZE_INPUTS (bool): Whether to normalize the inputs.
            shift (float array): Shift values applied during normalization to center the input range in 0.
            state_norm_arr (float array): Normalization factors for the state inputs.
            remapped_indices (list): List of indices for remapping input angles.
            cos_element (list): List of indices of elements to be encoded as cosines.
            sin_element (list): List of indices of elements to be encoded as sines.
            features_critic (list): List of layer sizes for the critic MLP.
    """
    conf: Any
    w0: float = 1.0
    c: float = 6.0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Normalize input
        if self.conf.NORMALIZE_INPUTS:
            x = (x + self.conf.shift) / self.conf.state_norm_arr

        # Optional angle remapping (to [cos, sin])
        if self.conf.remapped_indices is not None:
            if x.ndim > 1:
                x = x[:,self.conf.remapped_indices]
                x = x.at[:,self.conf.cos_element].set(jnp.cos(x[:,self.conf.cos_element]))
                x = x.at[:,self.conf.sin_element].set(jnp.sin(x[:,self.conf.sin_element]))
            else:
                x = x[self.conf.remapped_indices]
                x = x.at[self.conf.cos_element].set(jnp.cos(x[self.conf.cos_element]))
                x = x.at[self.conf.sin_element].set(jnp.sin(x[self.conf.sin_element]))

        # Feedforward layers
        for (i,feat) in enumerate(self.conf.features_critic[:-1]):
            if i == 0:
                x = SinusoidalRepresentationDense(features=feat, w0=self.w0, c=self.c, first_layer=True)(x)
            else:
                x = SinusoidalRepresentationDense(features=feat, w0=self.w0, c=self.c)(x)
        
        # Output layer (no activation)
        x = nn.Dense(self.conf.features_critic[-1], kernel_init=glorot_uniform(), bias_init=zeros)(x)

        return x

def create_networks(conf, seed=0):
    """Create and initialize the Actor, Critic, and STD-Critic networks."""
    key1, key_actor, key_critic, key_std_critic = jax.random.split(jax.random.PRNGKey(seed), 4)
    x_dummy = jax.random.normal(key1, (conf.nb_state,))
    
    # Initialize models
    actor = ActorMLP(conf)
    critic = CriticSineMLP(conf)
    std_critic = CriticSineMLP(conf)

    # Initialize training states
    actor_state = TrainStateA.create(
        apply_fn=actor.apply,
        params=actor.init(key_actor, x_dummy),
        prev_params=actor.init(key_actor, x_dummy),
        tx=optax.adamw(learning_rate=conf.ACTOR_LEARNING_RATE, weight_decay=conf.areg),
        )
    
    critic_state = TrainStateC.create(
        apply_fn=critic.apply,
        params=critic.init(key_critic, x_dummy),
        target_params=critic.init(key_critic, x_dummy),
        tx=optax.adamw(learning_rate=conf.CRITIC_LEARNING_RATE, weight_decay=conf.creg), 
        )
    
    std_critic_state = TrainState.create(
        apply_fn=std_critic.apply,
        params=std_critic.init(key_std_critic, x_dummy),
        tx=optax.adamw(learning_rate=conf.STD_CRITIC_LEARNING_RATE, weight_decay=1e-1),
        )

    return actor, critic, std_critic, actor_state, critic_state, std_critic_state

def save_weights(NNs_path, params, N_try, name='final'):
    """Save network parameters as a pickle file."""
    with open(NNs_path+"/N_try_{}/actor_{}.pkl".format(N_try,name), 'wb') as f:
        pickle.dump(params, f)
        
def load_weights(NNs_path, N_try, name='final'):
    """Load network parameters from a pickle file."""
    with open(NNs_path+"/N_try_{}/actor_{}.pkl".format(N_try,name), 'rb') as f:
        params = pickle.load(f)
    return params

@jax.jit
def custom_log(input):
    """Custom symmetric log function."""
    safe_input = jnp.where(input > 0, input + 1e-7, 1.)
    return jnp.where(input > 0, jnp.log(safe_input + 1), -jnp.log(jnp.where(input < 0, -input + 1e-7, 1.) + 1))
