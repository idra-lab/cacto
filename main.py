import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import jax
os.environ['XLA_FLAGS'] = (
    "--xla_backend_extra_options="
    "xla_cpu_disable_new_fusion_emitters=true"
)
import sys
import time
import shutil
import random
import argparse
import importlib
import jax.numpy as jnp
from functools import partial
from optax import incremental_update
from flax.training.train_state import TrainState

from plot_utils import PLOT
from TO import TO_JAX 
from replay_buffer import init_buffer, buffer_add, buffer_sample

from NN_jax import create_networks, save_weights, custom_log, TrainStateC, TrainStateA

jax.config.update("jax_enable_x64", False)

def parse_args():
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                           type=int,   default=0,                                    
                        help="Test number")
    
    parser.add_argument('--seed',                             type=int,   default=0,                                    
                        help="random and tf.random seed")

    parser.add_argument('--system-id',                        type=str,   default='single_integrator',
                        choices=["single_integrator", "double_integrator", "car", "manipulator", "aliengo", "reacher"],
                        help="System-id (single_integrator, double_integrator, car, manipulator, aliengo")

    parser.add_argument('--GPU-device',                       type=str,  default="1",
                        help="GPU device to use (set to None to use CPU)")
    
    parser.add_argument('--plot-flag',                        type=int,  default=0,
                        choices=[1, 0],
                        help="Flag to plot results")
    
    parser.add_argument('--w-S',                              type=float, default=1e-3,
                        help="Sobolev training - weight of the value related error (higher means more importance given to the value related error)")

    parser.add_argument('--BICSf',                            type=float, default=0,
                        help="BICS factor - percentage of the initial TO states sampled from the second cycle onwards")
    
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args




if __name__ == '__main__':
    args = parse_args()
    
    #########################################################
    ###############           Input           ###############
    #########################################################
    N_try     = args['test_n']

    if args['seed'] == None:
        seed = random.randint(1,100000)
    else:
        seed = args['seed']
    random.seed(seed)

    system_id = args['system_id'] 
    
    GPU_device = args['GPU_device']  
    if GPU_device is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

    plot_flag = args['plot_flag']

    w_S = args['w_S']

    BICS_factor = args['BICSf']
    #########################################################



    # Import configuration file and environment file
    system_map = {
        'single_integrator': ('conf_single_integrator', 'SingleIntegrator_CAMS'),
        'double_integrator': ('conf_double_integrator', 'DoubleIntegrator_CAMS'),
        'car':               ('conf_car', 'Car_CAMS'),
        'manipulator':       ('conf_manipulator', 'Manipulator_CAMS'),
        'aliengo':           ('conf_aliengo', 'AlienGo_CAMS'),
        'reacher':           ('conf_reacher', 'Reacher_CAMS')
    }
    try:
        conf_module, env_TO_class = system_map[system_id]
        conf = importlib.import_module(conf_module)
        Environment_TO = getattr(importlib.import_module('environment_TO'), env_TO_class)
    except KeyError:
        print('System {} not found'.format(system_id))
        sys.exit()

    # Create folders to store the results and the trained NNs
    for path in conf.path_list:
        os.makedirs(path + '/N_try_{}'.format(N_try), exist_ok=True)
    os.makedirs(conf.Config_path, exist_ok=True)

    # Save configuration
    params = [p for p in conf.__dict__ if not p.startswith("__")]
    with open(conf.Config_path + '/config{}.txt'.format(N_try), 'w') as f:
        for p in params:
            f.write('{} = {}\n'.format(p, conf.__dict__[p]))
        f.write('Seed = {}\n'.format(seed))
        f.write('w_S = {}'.format(w_S))

    # Save code
    shutil.copy('{}.py'.format(conf_module), conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try))
    with open(conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try), 'a') as f:
        f.write('\n\n# {}'.format(args))
    
    for file in os.listdir("./"):
        if file.endswith(".py"):
            shutil.copy(os.path.join("./", file), os.path.join(conf.Code_path + '/N_try_{}'.format(N_try), file))

    # Create empty txt file in Log_path to store the test info
    open(conf.Log_path + '/info.txt', 'a').close()



    ### Create instances of the main classes ###
    env_TO = Environment_TO('running_model', conf)          # Initialize the environment
    TrOpU = TO_JAX(env_TO, conf)                            # Initialize Trajectory Optimization class for uniform initialization
    if BICS_factor > 0:
        TrOpB = TO_JAX(env_TO, conf)                        # Initialize Trajectory Optimization class for biased initialization (if BICS is enabled)
    TrOpPlot =  TO_JAX(env_TO, conf)                        # Initialize Trajectory Optimization class for plotting

    buffer = init_buffer(conf)                              # Initialize replay buffer for RL training
    
    plot_fun = PLOT(N_try, env_TO, TrOpPlot, conf)          # Initialize plotting utilities                                                                   
    
    
    @jax.jit
    def update_critic(
            critic_state: TrainStateC,
            observations: jnp.ndarray,
            next_observations: jnp.ndarray,
            rewards: jnp.ndarray,
            dV_values: jnp.ndarray,
            terminations: jnp.ndarray,
            ):
        """Update the critic network parameters."""
        
        # Compute target critic values (bootstrapped from next state)
        critic_next_target = critic.apply(critic_state.target_params, next_observations).reshape(-1)
        next_critic_value = (rewards.reshape(-1) + (1 - terminations).reshape(-1) * (critic_next_target)).reshape(-1)
        def mse_loss(params):
            """Mean squared error loss with gradient matching."""
            critic_values = critic.apply(params, observations).squeeze()
            dcritic_values = jax.vmap(jax.grad(lambda s: critic.apply(params, s).squeeze()))(observations)

            return (w_S*(critic_values - next_critic_value)**2 + jnp.mean((custom_log(dcritic_values[:,:conf.nb_state-1]) - custom_log(dV_values[:,:conf.nb_state-1]))**2, axis=1).reshape(-1)).mean()

        # Compute loss and gradients
        critic_loss_value, grads = jax.value_and_grad(mse_loss)(critic_state.params)

        # Apply gradient update
        critic_state = critic_state.apply_gradients(grads=grads)

        # Soft update of target network parameters
        critic_state = critic_state.replace(
            target_params=incremental_update(critic_state.params, critic_state.target_params, conf.UPDATE_RATE)
        )
        
        return critic_state, critic_loss_value
    
    @jax.jit
    def update_std_critic(
            std_critic_state: TrainState,
            critic_state: TrainState,
            observations: jnp.ndarray,
            next_observations: jnp.ndarray,
            rewards: jnp.ndarray,
            terminations: jnp.ndarray,
            ):
        """Update the standard deviation critic network."""
        
        # Compute critic target values
        critic_next_target = critic.apply(critic_state.target_params, next_observations).reshape(-1)
        
        # Compute current critic error
        reference_values = (rewards.reshape(-1) + (1 - terminations).reshape(-1) * (critic_next_target)).reshape(-1)
        critic_values = critic.apply(critic_state.params, observations).squeeze()
        critic_error = (reference_values - critic_values)**2
        
        def mse_loss(params):
            # Compute std critic values
            std_critic_values = std_critic.apply(params, observations).squeeze()
            exp_std_critic_values = jnp.exp(std_critic_values)
            
            return (std_critic_values + critic_error / exp_std_critic_values).mean()

        # Compute loss and gradients
        std_critic_loss_value, grads = jax.value_and_grad(mse_loss)(std_critic_state.params)
        
        # Apply gradient update
        std_critic_state = std_critic_state.apply_gradients(grads=grads)

        return std_critic_state, std_critic_loss_value
    
    @jax.jit
    def update_actor(
        actor_state: TrainStateA,
        critic_state: TrainStateC,
        observations: jnp.ndarray,
        ):
        """Update the actor network parameters."""
        
        # Store current parameters
        actor_state = actor_state.replace(prev_params=actor_state.params)

        # Discrete time update
        tau = actor.apply(actor_state.params, observations)
        state_next = jax.vmap(env_TO.dynamics_tau_func_wrapped)(observations, tau)
        dcritic_dstate_next = jax.vmap(jax.grad(lambda s: critic.apply(critic_state.params, s).squeeze()))(state_next)[:, None, :]
        ddyn_da = jax.vmap(jax.jacrev(env_TO.dynamics_tau_func_wrapped, argnums=1))(observations, tau)/conf.state_norm_arr[:,None]
        dcritic_da = (dcritic_dstate_next @ ddyn_da).squeeze()[:, None, :]

        def actor_loss(params):
            # Compute actor loss
            tau = actor.apply(params, observations)
            return ((dcritic_da @ tau[:, :, None]).squeeze() + jax.vmap(env_TO.cost_tau_func_wrapped)(observations, tau).squeeze()).mean()
        
        # Compute loss and gradients
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)

        # Apply gradient update
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, actor_loss_value
    
    jax.jit
    def sample_ICS(n, key):
        """Sample uniformly initial conditions"""
        if system_id == 'reacher':
            subkey, subkey_dist, subkey_ang, subkey_other = jax.random.split(key, 4)
            dist = jax.random.uniform(subkey_dist, shape=(n, 1), minval=conf.x_init_min[4], maxval=conf.x_init_max[4])
            ang = jax.random.uniform(subkey_ang, shape=(n, 1), minval=conf.x_init_min[5], maxval=conf.x_init_max[5])
            init_rand_state_t = jax.random.uniform(subkey_other, shape=(n, conf.nb_state), minval=conf.x_init_min, maxval=conf.x_init_max)
            
            init_rand_state_t = init_rand_state_t.at[:,4].set((dist * jnp.cos(ang)).squeeze())
            init_rand_state_t = init_rand_state_t.at[:,5].set((dist * jnp.sin(ang)).squeeze())
            p_ee = jax.vmap(env_TO.p_ee_jax_wrapped)(init_rand_state_t)
            init_rand_state_t = init_rand_state_t.at[:,6:8].set(p_ee[:,:2]-init_rand_state_t[:,4:6])
            init_rand_t = jnp.zeros((n,))
            
        elif system_id == 'aliengo':
            subkey, subkey_wsn, subkey_wsp, subkey_xcom, subkey_obs = jax.random.split(key, 5)
            init_rand_state_t = jax.random.uniform(subkey, shape=(n, conf.nb_state), minval=conf.x_init_min, maxval=conf.x_init_max)
            wall_state_pos = jax.random.uniform(subkey_wsp, shape=(n,2), minval=1*jnp.ones(2), maxval=(conf.room_size-1)*jnp.ones(2))
            wall_state_neg = jax.random.uniform(subkey_wsn, shape=(n,2), minval=-conf.room_size*jnp.ones(2)+wall_state_pos, maxval=-1*jnp.ones(2))
            xCoM_state = jax.random.uniform(subkey_xcom, shape=(n,2), minval=wall_state_neg, maxval=wall_state_pos)
            obstacle_state = jax.random.uniform(subkey_obs, shape=(n,2), minval=wall_state_neg, maxval=wall_state_pos)
            init_rand_state_t = init_rand_state_t.at[:, 4:6].set(xCoM_state)
            init_rand_state_t = init_rand_state_t.at[:, 8:10].set(obstacle_state)
            init_rand_state_t = init_rand_state_t.at[:, 10:12].set(wall_state_pos)
            init_rand_state_t = init_rand_state_t.at[:, 12:14].set(wall_state_neg)

            subkey, subkey_wsn, subkey_wsp, subkey_xcom, subkey_obs = jax.random.split(subkey, 5)
            wall_state_neg = jax.random.uniform(subkey_wsn, shape=(n//2, 2), minval=-conf.room_size * jnp.ones(2), maxval=-1 * jnp.ones(2))
            wall_state_pos = jax.random.uniform(subkey_wsp, shape=(n//2, 2), minval=1 * jnp.ones(2), maxval=(conf.room_size - 1) * jnp.ones(2) + wall_state_neg)
            xCoM_state = jax.random.uniform(subkey_xcom, shape=(n//2, 2), minval=wall_state_neg, maxval=wall_state_pos)
            obstacle_state = jax.random.uniform(subkey_obs, shape=(n//2, 2), minval=wall_state_neg, maxval=wall_state_pos)
            init_rand_state_t = init_rand_state_t.at[-n//2:, 4:6].set(xCoM_state)
            init_rand_state_t = init_rand_state_t.at[-n//2:, 8:10].set(obstacle_state)
            init_rand_state_t = init_rand_state_t.at[-n//2:, 10:12].set(wall_state_pos)
            init_rand_state_t = init_rand_state_t.at[-n//2:, 12:14].set(wall_state_neg)

            init_rand_state_t = init_rand_state_t.at[:,14].set(jnp.sqrt((init_rand_state_t[:,4] - init_rand_state_t[:,8])**2 + (init_rand_state_t[:,5] - init_rand_state_t[:,9])**2))
            init_rand_state_t = init_rand_state_t.at[:,15].set(jnp.sqrt((init_rand_state_t[:,4])**2 + (init_rand_state_t[:,5])**2))

            init_rand_state_t = init_rand_state_t.at[:, 16:17].set(0)#1.000*jax.random.randint(subkey, shape=(n,1), minval=0, maxval=2))
            init_rand_t = jnp.zeros((n,))

        elif system_id == 'car':
            subkey, subkey_other = jax.random.split(key, 2)
            init_rand_state_t = jax.random.uniform(subkey_other, shape=(n, conf.nb_state), minval=conf.x_init_min, maxval=conf.x_init_max)
            sin = jnp.sqrt(1-jnp.square(init_rand_state_t[:,2]))
            init_rand_state_t = init_rand_state_t.at[:,3].set(sin)
            init_rand_t = jnp.zeros((n,))

        else:
            init_rand_state_t = jax.random.uniform(key, shape=(n, conf.nb_state), minval=conf.x_init_min, maxval=conf.x_init_max)
            init_rand_t = jnp.zeros((n,))

        return init_rand_state_t, init_rand_t

    def create_unif_TO_init(n_UICS, key):
        ''' Create n uniformely distributed ICS '''
        # Create uniformly distributed initial conditions (ICS)
        init_rand_state_t, init_rand_t = sample_ICS(n_UICS, key)

        return init_rand_state_t, init_rand_t
    
    @jax.jit
    def evaluate_ics(init_rand_state_t, std_critic_state, actor_state, critic_state):
        """Evaluate initial conditions (ICS) based on critic uncertainty and actor optimality."""
        # Check feasibility of each initial condition
        mask_unfeas = jax.vmap(env_TO.check_feasible_wrapped)(init_rand_state_t) != 1

        # Compute critic uncertainty (C)
        std_values_C = jnp.exp(jax.vmap(lambda x: std_critic.apply(std_critic_state.params, x).squeeze())(init_rand_state_t)) - mask_unfeas*jnp.inf

        # Sort states by s=C
        idx = jnp.argsort(std_values_C)

        return init_rand_state_t[idx]
    
    def create_biased_TO_init(n_BICS, std_critic_state, actor_state, critic_state, key, factor=10):
        """Create biased initial conditions (ICS)"""
        # Sample an overset of random initial conditions
        init_rand_state_t, _ = sample_ICS(n_BICS*factor, key)
        
        # Evaluate and sort them based on s-based metric
        sorted_init_rand_state_t = evaluate_ics(init_rand_state_t, std_critic_state, actor_state, critic_state)

        # Select n_BICS states with highest combined score
        return sorted_init_rand_state_t[-n_BICS:, :], sorted_init_rand_state_t[-n_BICS:, -1]

    def one_update_step(carry, step_idx):
        (key, actor_state, critic_state, buffer, n_updates) = carry

        # Mask: whether this step is active
        do_update = step_idx < n_updates

        def do_update_fn(carry):
            (key, actor_state, critic_state, buffer, n_updates) = carry

            key, subkey = jax.random.split(key)

            state_b, rtg_b, state_next_b, dVdx_b, d_b, w_b = buffer_sample(buffer, subkey, batch_size=conf.BATCH_SIZE, n_sample=1)

            state_b, state_next_b, rtg_b, dVdx_b, d_b = state_b[0], state_next_b[0], rtg_b[0], dVdx_b[0], d_b[0]

            critic_state, critic_loss = update_critic( critic_state, state_b, state_next_b, rtg_b, dVdx_b, d_b)

            actor_state, actor_loss = update_actor(actor_state, critic_state, state_b)

            return (key, actor_state, critic_state, buffer, n_updates), (critic_loss, actor_loss, actor_state.params)

        def skip_update_fn(carry):
            key, actor_state, critic_state, buffer, n_updates = carry
            return carry, (0.0, 0.0, actor_state.params)

        new_carry, losses = jax.lax.cond(do_update, do_update_fn, skip_update_fn, carry)

        return new_carry, losses

    @partial(jax.jit, static_argnames=("MAX_UPDATES",))
    def run_all_updates_jit(
        key,
        actor_state,
        critic_state,
        buffer,
        n_updates,
        MAX_UPDATES,
    ):
        init = (key, actor_state, critic_state, buffer, n_updates)

        steps = jnp.arange(MAX_UPDATES)

        final_carry, (critic_losses, actor_losses, actor_state_params) = jax.lax.scan(one_update_step, init, steps)

        return final_carry, (critic_losses, actor_losses, actor_state_params)
    
    def one_stdc_update_step(carry, step_idx):
        (key, std_critic_state, critic_state, buffer, n_updates) = carry

        # Mask: whether this step is active
        do_update = step_idx < n_updates

        def do_update_fn(carry):
            (key, std_critic_state, critic_state, buffer, n_updates) = carry

            key, subkey = jax.random.split(key)

            state_b, rtg_b, state_next_b, dVdx_b, d_b, w_b = buffer_sample(buffer, subkey, batch_size=conf.BATCH_SIZE, n_sample=1)

            state_b, state_next_b, rtg_b, dVdx_b, d_b = state_b[0], state_next_b[0], rtg_b[0], dVdx_b[0], d_b[0]

            std_critic_state, std_critic_loss = update_std_critic(std_critic_state, critic_state, state_b, state_next_b, rtg_b, d_b)

            return (key, std_critic_state, critic_state, buffer, n_updates), std_critic_loss

        def skip_update_fn(carry):
            return carry, 0.0

        new_carry, losses = jax.lax.cond(do_update, do_update_fn, skip_update_fn, carry)

        return new_carry, losses


    @partial(jax.jit, static_argnames=("MAX_UPDATES",))
    def run_all_stdc_updates_jit(
        key,
        std_critic_state,
        critic_state,
        buffer,
        n_updates,
        MAX_UPDATES,
    ):
        init = (key, std_critic_state, critic_state, buffer, n_updates)

        # Static loop indices (enable a single compilation)
        steps = jnp.arange(MAX_UPDATES)

        final_carry, losses = jax.lax.scan(one_stdc_update_step, init, steps)

        return final_carry, losses





    ### START TRAINING ###
    if conf.profile:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    time_start_init_all = time.time()
    
    # Create random key
    key = jax.random.PRNGKey(seed)  

    # Initialize networks
    actor, critic, std_critic, actor_state, critic_state, std_critic_state = create_networks(conf, seed=seed)  
    save_weights(conf.NNs_path, actor_state.params, N_try, 0)

    # Plot reward function
    if plot_flag:
        plot_fun.plot_traj_from_ICS_jax(jnp.array(conf.init_states_sim), actor.apply, actor_state.params, sub_name=0, steps=conf.NSTEPS, init=0, psd_delta=1e-4)

    # Initialize step counter
    update_step_counter = 0

    # Initialize TO
    start = time.time()
    n_TO_U = conf.EP_UPDATE
    MAX_UPDATES = max(conf.UPDATE_LOOPS)

    TrOpU.TO_System_Solve(jnp.zeros((n_TO_U, conf.nb_state)), jnp.zeros((n_TO_U, conf.NSTEPS, conf.nb_action)), jnp.ones(n_TO_U), maxiter=conf.maxiter, psd_delta=1e-4)
    TrOpU.create_TO_ws(jnp.zeros((n_TO_U, conf.nb_state)), actor.apply, actor_state.params, init=update_step_counter)         
    if BICS_factor > 0:
        n_TO_B = int(n_TO_U*BICS_factor/conf.quant_maxiter)
        TrOpB.TO_System_Solve(jnp.zeros((n_TO_B, conf.nb_state)), jnp.zeros((n_TO_B, conf.NSTEPS, conf.nb_action)), jnp.ones(n_TO_B), maxiter=conf.maxiter_ws, psd_delta=1e-4)
        TrOpB.create_TO_ws(jnp.zeros((n_TO_B, conf.nb_state)), actor.apply, actor_state.params, init=1)           
    print('Elapsed time TO init: ', time.time()-start)

    # Start training
    time_start_init = time.time()
    for ep in range(conf.NLOOPS): 
        print('#### EPISODE {} ####'.format(ep))

        if plot_flag:
            # Plot trajectories with naive warmstart (stabilize controller in conf)
            if ep == 0:
                plot_fun.plot_traj_from_ICS_jax(jnp.array(conf.init_states_sim), actor.apply, actor_state.params, sub_name=0, steps=conf.NSTEPS, init=0, psd_delta=1e-4) 

        # Create random keys
        key, subkey_create_init, subkey_sample_c, subkey_sample_a, subkey_sample_stdc = jax.random.split(key, 5)

        # Create initial states (ICS)
        time_start = time.time()
        if ep > 0 and BICS_factor > 0:
            init_rand_state, init_rand_t = create_biased_TO_init(n_TO_B, std_critic_state, actor_state, critic_state, subkey_create_init, factor=10)
            max_iter = conf.maxiter_ws
        else:
            init_rand_state, init_rand_t = create_unif_TO_init(n_TO_U, subkey_create_init)
            max_iter = conf.maxiter
        if plot_flag:
            # Plot ICS samples
            plot_fun.plot_ICS(jnp.array(init_rand_state), name='ICS_{}'.format(ep))

        # Compute horizon length based on sampled time
        NSTEPS_SH = conf.NSTEPS - jnp.floor(init_rand_t / conf.dt)

        TrOp = TrOpB if (ep > 0 and BICS_factor > 0) else TrOpU

        # Create warmstart for the TO problem
        time_start = time.time()
        _, init_TO_controls, success_init_flag = TrOp.create_TO_ws(init_rand_state, actor.apply, actor_state.params, init=min(1,update_step_counter))           
        print('Create init TO : ', time.time()-time_start)

        # Solve the TO problem
        time_start = time.time()
        TO_states, TO_controls, TO_ee_pos_arr, TO_step_cost, sf, dVdx = TrOp.TO_System_Solve(init_rand_state, init_TO_controls, NSTEPS_SH, maxiter=max_iter, psd_delta=1e-4)
        print('Solve TO ({:.1f}) : '.format(sum(sf != 0)), time.time()-time_start)

        # Postprocess TO data
        time_start = time.time()
        state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, RL_ee_pos_arr, dVdx_arr = jax.vmap(TrOp.postprocess_TO_data)(TO_states, TO_step_cost, TO_ee_pos_arr, dVdx)
        print('Postprocess TO : ', time.time()-time_start)

        # Filter valid transition (remove unsuccess TO and out of the horizon transitions)
        time_start = time.time()
        success_mask = sf != 0
        indices = jnp.arange(conf.NSTEPS+1)
        valid_mask = indices[None, :] < NSTEPS_SH[:, None]
        mask = success_mask[:, None] & valid_mask
        data = [state_arr[mask].reshape(-1,conf.nb_state), partial_reward_to_go_arr[mask].reshape(-1,1), state_next_rollout_arr[mask].reshape(-1,conf.nb_state), dVdx_arr[mask].reshape(-1,conf.nb_state), done_arr[mask].reshape(-1,1)]
    
        # Update buffer
        buffer = buffer_add(buffer, data)
        print('Store data     : ', time.time()-time_start)
        
        # Update actor-critic networks
        remaining = conf.NUPDATES - update_step_counter
        n_updates = min(conf.UPDATE_LOOPS[ep], max(remaining, 0))

        t_preup = time.time() 
        if ep > 0:
            offset = update_step_counter
        else:
            offset = 0
        (key, actor_state, critic_state, buffer, _), (critic_losses, actor_losses, actor_state_params) = run_all_updates_jit(key, actor_state, critic_state, buffer, n_updates, MAX_UPDATES=MAX_UPDATES)
        update_step_counter += n_updates
        t_ac = time.time() - t_preup

        t_s = time.time()
        for s in range(int(n_updates)):
            if (offset+s) % conf.save_interval == 0:
                print((offset+s), " : ", t_preup - time_start_init + s*t_ac/n_updates)
                save_weights(conf.NNs_path, jax.tree.map(lambda x: x[s], actor_state_params), N_try, int(offset+s))
                #save_weights(conf.NNs_path,jax.tree_map(lambda x: x[s], actor_state_params), N_try, int(offset+s))
        print('Saving time    : ', time.time()-t_s)
        
        # Update std-critic network
        t_stdc = 0 
        if BICS_factor > 0 and update_step_counter < conf.NUPDATES:
            t_s = time.time()
            (key, std_critic_state, critic_state, buffer, _), std_critic_losses = run_all_stdc_updates_jit(key, std_critic_state, critic_state, buffer, n_updates, MAX_UPDATES=MAX_UPDATES)
            t_stdc += time.time()-t_s
        
        print(ep, 'TO problem solved: {}'.format(jnp.sum(success_mask)), 'ActorCritic time: {:3f} '.format(t_ac), 'STDCritic time: {:3f}'.format(t_stdc))
    
        if plot_flag:    
            # Plot rollouts and state and control trajectories
            print("System: {} - N_try = {}".format(system_id, N_try))
            plot_fun.plot_Critic_Value_function(critic.apply, critic_state.params, ep, system_id, name='V')
            plot_fun.plot_traj_from_ICS_jax(jnp.array(conf.init_states_sim), actor.apply, actor_state.params, sub_name=ep, steps=conf.NSTEPS, maxiter=conf.maxiter, psd_delta=1e-4)

        if update_step_counter >= conf.NUPDATES:
            break

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start_init)
    print('Elapsed time all: ', time_end-time_start_init_all)

    if conf.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
