import jax
import flax
import optax
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from typing import Callable, Any
from jax.nn.initializers import glorot_uniform, zeros
from flax.training.train_state import TrainState

# Define Siren layer
class Sine(nn.Module):
    w0: float = 1.0
    
    @nn.compact
    def __call__(self, x):
        return jnp.sin(self.w0 * x)

def siren_init(w0: float = 1.0, c: float = 6.0, dtype: Any = jnp.float_):
    def init_fn(key, shape, dtype=dtype):
        fan_in = shape[0] if len(shape) == 2 else jnp.prod(shape[:-1])
        limit = jnp.sqrt(c) / w0
        return random.uniform(key, shape, dtype, minval=-limit/fan_in, maxval=limit/fan_in)
    return init_fn

def siren_first_layer_init(scale: float = 1.0, dtype: Any = jnp.float_):
    def init_fn(key, shape, dtype=dtype):
        fan_in = shape[0] if len(shape) == 2 else jnp.prod(shape[:-1])
        limit = scale / max(1.0, fan_in)
        return random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return init_fn

def he_uniform_bias(n):
    """Versione semplificata di he_uniform per bias (vettori 1D)"""
    def init(key, shape, dtype=jnp.float32):
        limit = jnp.sqrt(6.0 / n)
        return jax.random.uniform(
            key, shape, dtype, 
            minval=-limit, 
            maxval=limit
        )
    return init

class SinusoidalRepresentationDense(nn.Module):
    features: int
    w0: float = 1.0
    c: float = 6.0
    use_bias: bool = True
    kernel_init: Callable = None
    #bias_init: Callable = nn.initializers.zeros
    first_layer: bool = False
    
    @nn.compact
    def __call__(self, inputs):
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
            bias_init=he_uniform_bias(n=self.features) #self.bias_init
        )(inputs)
        
        return Sine(w0=self.w0)(x)
    
class ActorMLP(nn.Module):
    conf: Any
    @nn.compact
    def __call__(self, x):
        if self.conf.NORMALIZE_INPUTS:
            x = (x + self.conf.shift) / self.conf.state_norm_arr
        if self.conf.system_id == 'manipulator' and self.conf.remap_angle:
            if x.ndim > 1:
                cos_angles = jnp.cos(x[:, :3])
                sin_angles = jnp.sin(x[:, :3])
                x = jnp.concatenate([cos_angles[:,0].reshape(-1,1), sin_angles[:,0].reshape(-1,1), cos_angles[:,1].reshape(-1,1), sin_angles[:,1].reshape(-1,1), cos_angles[:,2].reshape(-1,1), sin_angles[:,2].reshape(-1,1), x[:, 3:]], axis=1)
            else:
                cos_angles = jnp.cos(x[:3])
                sin_angles = jnp.sin(x[:3])
                x = jnp.hstack([cos_angles[0], sin_angles[0], cos_angles[1], sin_angles[1], cos_angles[2], sin_angles[2], x[3], x[4], x[5], x[6]])
        for feat in self.conf.features_actor[:-1]:
            x = nn.Dense(feat, kernel_init=glorot_uniform(), bias_init=zeros)(x)
            x = nn.leaky_relu(x, negative_slope=0.3)
        x = nn.Dense(self.conf.features_actor[-1], kernel_init=glorot_uniform(), bias_init=zeros)(x)
        x = nn.tanh(x)*self.conf.u_max
        return x



class CriticSineMLP(nn.Module):
    conf: Any
    w0: float = 1.0
    c: float = 6.0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.conf.NORMALIZE_INPUTS:
            x = (x + self.conf.shift) / self.conf.state_norm_arr
        if self.conf.system_id == 'manipulator' and self.conf.remap_angle:
            if x.ndim > 1:
                cos_angles = jnp.cos(x[:, :3])
                sin_angles = jnp.sin(x[:, :3])
                x = jnp.concatenate([cos_angles[:,0].reshape(-1,1), sin_angles[:,0].reshape(-1,1), cos_angles[:,1].reshape(-1,1), sin_angles[:,1].reshape(-1,1), cos_angles[:,2].reshape(-1,1), sin_angles[:,2].reshape(-1,1), x[:, 3:]], axis=1)
            else:
                cos_angles = jnp.cos(x[:3])
                sin_angles = jnp.sin(x[:3])
                x = jnp.hstack([cos_angles[0], sin_angles[0], cos_angles[1], sin_angles[1], cos_angles[2], sin_angles[2], x[3], x[4], x[5], x[6]])
    
        for (i,feat) in enumerate(self.conf.features_critic[:-1]):
            if i == 0:
                x = SinusoidalRepresentationDense(features=feat, w0=self.w0, c=self.c, first_layer=True)(x)
            else:
                x = SinusoidalRepresentationDense(features=feat, w0=self.w0, c=self.c)(x)
        x = nn.Dense(self.conf.features_critic[-1], kernel_init=glorot_uniform(), bias_init=zeros)(x)
        return x



class CriticEluMLP(nn.Module):
    conf: Any
    @nn.compact
    def __call__(self, x):
        if self.conf.NORMALIZE_INPUTS:
            x = (x + self.conf.shift) / self.conf.state_norm_arr
        if self.conf.system_id == 'manipulator' and self.conf.remap_angle:
            if x.ndim > 1:
                cos_angles = jnp.cos(x[:, :3])
                sin_angles = jnp.sin(x[:, :3])
                x = jnp.concatenate([cos_angles[:,0].reshape(-1,1), sin_angles[:,0].reshape(-1,1), cos_angles[:,1].reshape(-1,1), sin_angles[:,1].reshape(-1,1), cos_angles[:,2].reshape(-1,1), sin_angles[:,2].reshape(-1,1), x[:, 3:]], axis=1)
            else:
                cos_angles = jnp.cos(x[:3])
                sin_angles = jnp.sin(x[:3])
                x = jnp.hstack([cos_angles[0], sin_angles[0], cos_angles[1], sin_angles[1], cos_angles[2], sin_angles[2], x[3], x[4], x[5], x[6]])
        for feat in self.conf.features_critic[:-1]:
            x = nn.elu(nn.Dense(feat, kernel_init=glorot_uniform(), bias_init=zeros)(x))
        x = nn.Dense(self.conf.features_critic[-1], kernel_init=glorot_uniform(), bias_init=zeros)(x)  # No activation on last layer
        return x



class TrainStateC(TrainState):
    target_params: flax.core.FrozenDict



def create_networks(conf, seed=0):
    ''' Create NNs and initialize optimizer'''
    key1, key_actor, key_critic, key_std_critic = jax.random.split(jax.random.PRNGKey(seed), 4)
    x_dummy = jax.random.normal(key1, (conf.nb_state,))
    
    # Initialize models
    actor = ActorMLP(conf)
    critic = CriticSineMLP(conf)
    std_critic = CriticSineMLP(conf)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(key_actor, x_dummy),
        tx=optax.adam(learning_rate=conf.ACTOR_LEARNING_RATE),
        )
    
    critic_state = TrainStateC.create(
        apply_fn=critic.apply,
        params=critic.init(key_critic, x_dummy),
        target_params=critic.init(key_critic, x_dummy),
        tx=optax.adam(learning_rate=conf.CRITIC_LEARNING_RATE),
        )
    
    std_critic_state = TrainState.create(
        apply_fn=std_critic.apply,
        params=std_critic.init(key_std_critic, x_dummy),
        tx=optax.adam(learning_rate=conf.STD_CRITIC_LEARNING_RATE),
        )
    
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    std_critic.apply = jax.jit(std_critic.apply)

    return actor, critic, std_critic, actor_state, critic_state, std_critic_state

def save_weights(NNs_path, params, N_try, update_step_counter='final'):
    ''' Save NN weights '''
    with open(NNs_path+"/N_try_{}/actor_{}.pkl".format(N_try,update_step_counter), 'wb') as f:
        pickle.dump(params, f)

def get_regularization(model_params, l1_weight=0.0, l2_weight=0.0):
    ''' Compute L1 and L2 regularization for the parameters of the model '''
    reg_loss = 0.0
        
    def sum_l2(tree):
        sum_squared = jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2),
            tree,
            initializer=0.0
        )
        return sum_squared
        
    def sum_l1(tree):
        sum_abs = jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.abs(y),
            tree,
            initializer=0.0
        )
        return sum_abs
        
    weight_params = {}
    bias_params = {}
        
    for layer_name, layer_params in model_params.items():
        if 'kernel' in layer_params:
            weight_params[layer_name] = {'kernel': layer_params['kernel']}
        if 'bias' in layer_params:
            bias_params[layer_name] = {'bias': layer_params['bias']}
        
    reg_loss += sum_l2(weight_params) * l2_weight + sum_l1(weight_params) * l1_weight
    
    reg_loss += sum_l2(bias_params) * l2_weight + sum_l1(bias_params) * l1_weight
    
    return reg_loss

@jax.jit
def custom_log(input):
    ''' Custom symmetric log function '''
    safe_input = jnp.where(input > 0, input + 1e-7, 1.)  # Ensure finite adjoint
    return jnp.where(input > 0, jnp.log(safe_input + 1), -jnp.log(jnp.where(input < 0, -input + 1e-7, 1.) + 1))

def update_critic(
    critic_state: TrainState,
    observations: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    dV_values: np.ndarray,
    terminations: np.ndarray,
    w_S: float,
    critic: Any
    ):
    ''' Update critic network '''

    @jax.jit
    def update_critic_jit(
            critic_state: TrainState,
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            dV_values: np.ndarray,
            terminations: np.ndarray
            ):
        # Compute target values
        critic_next_target = critic.apply(critic_state.target_params, next_observations).reshape(-1)
        next_critic_value = (rewards.reshape(-1) + (1 - terminations).reshape(-1) * (critic_next_target)).reshape(-1)

        def mse_loss(params):
            # Compute critic values and gradients
            critic_values = critic.apply(params, observations).squeeze()
            dcritic_values = jax.vmap(jax.grad(lambda s: critic.apply(params, s).squeeze()))(observations)
            
            return (w_S*(critic_values - next_critic_value)** 2 + jnp.mean((custom_log(dcritic_values[:,:-1]) - custom_log(dV_values[:,:-1]))**2, axis=1).reshape(-1)).mean() #+ 1e-3 * kernel_reg + 1e-3 * bias_reg

        # Compute loss and gradients
        critic_loss_value, grads = jax.value_and_grad(mse_loss)(critic_state.params)

        # Update critic network
        critic_state = critic_state.apply_gradients(grads=grads)
        
        return critic_state, critic_loss_value
    
    return update_critic_jit(critic_state, observations, next_observations, rewards, dV_values, terminations)
    
def update_std_critic(
        std_critic_state: TrainState,
        critic_state: TrainState,
        observations: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        critic: Any,
        std_critic: Any
        ):
    ''' Update std critic network '''

    @jax.jit
    def update_std_critic_jit(
            std_critic_state: TrainState,
            critic_state: TrainState,
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
        ):
        # Compute target values
        critic_next_target = critic.apply(critic_state.target_params, next_observations).reshape(-1)

        # Compute critic error
        reference_values = (rewards.reshape(-1) + (1 - terminations).reshape(-1) * (critic_next_target)).reshape(-1)
        critic_values = critic.apply(critic_state.params, observations).squeeze()
        critic_error = (reference_values - critic_values)**2
    
        def mse_loss(params):
            # Compute std critic values
            std_critic_values = std_critic.apply(params, observations).squeeze()
            exp_std_critic_values = jnp.exp(std_critic_values)
        
            return (std_critic_values + critic_error / exp_std_critic_values).mean() #+ get_regularization(params, 0.1, 0.1)

        # Compute loss and gradients
        std_critic_loss_value, grads = jax.value_and_grad(mse_loss)(std_critic_state.params)

        # Update std critic network
        std_critic_state = std_critic_state.apply_gradients(grads=grads)

        return std_critic_state, std_critic_loss_value
    
    return update_std_critic_jit(std_critic_state, critic_state, observations, next_observations, rewards, terminations)
    
def update_actor(
    actor_state: TrainState,
    critic_state: TrainState,
    observations: np.ndarray,
    critic: Any,
    actor: Any,
    dynamics_func: Callable,
    cost_func: Callable,
    UPDATE_RATE: float,
    dt: float
    ):
    ''' Update actor network '''

    @jax.jit
    def update_actor_jit(
        actor_state: TrainState,
        critic_state: TrainState,
        observations: np.ndarray):

        def actor_loss(params):
            # Compute actions and the corresponding next states
            tau = actor.apply(params, observations)
            x_next =  jax.vmap(dynamics_func)(observations, tau)

            return (critic.apply(critic_state.params, x_next).squeeze() + jax.vmap(cost_func)(observations, tau).squeeze()).mean() + get_regularization(params, 0.01, 0.01)
        
        #tau = actor.apply(actor_state.params, observations)
        #dcritic_dstate = jax.vmap(jax.grad(lambda s: critic.apply(critic_state.params, s).squeeze()))(observations)[:, None, :]
        #ddyn_da = jax.vmap(jax.jacrev(dynamics_tau_func_so_nolist, argnums=1))(observations, tau)#.transpose(1,0,2)
        #dcritic_da = (dcritic_dstate @ ddyn_da/conf.dt*conf.dt).squeeze()[:, None, :]
        #def actor_loss_CT(params):
        #    tau = actor.apply(params, observations)
        #    return ((dcritic_da @ tau[:, :, None]).squeeze() + jax.vmap(cost_func_aug_nolist)(observations, tau).squeeze()).mean() + get_l2_regularization(params, 0.01) + get_l1_regularization(params, 0.01)
    
        # Compute loss and gradients
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)

        # Update actor network
        actor_state = actor_state.apply_gradients(grads=grads)

        # Update critic target network
        critic_state = critic_state.replace(
            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, UPDATE_RATE)
        )
        return actor_state, critic_state, actor_loss_value
    
    return update_actor_jit(actor_state, critic_state, observations)