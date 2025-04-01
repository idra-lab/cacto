import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import jax
import sys
import time
import shutil
import random
import argparse
import importlib
import numpy as np
import jax.numpy as jnp

from plot_utils import PLOT
from TO import TO_Casadi, TO_JAX 
from replay_buffer import ReplayBuffer_JAX

from NN_jax import create_networks, save_weights, get_regularization, custom_log

#from utils import *

def parse_args():
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                           type=int,   default=0,                                    
                        help="Test number")
    
    parser.add_argument('--seed',                             type=int,   default=0,                                    
                        help="random and tf.random seed")

    parser.add_argument('--system-id',                        type=str,   default='oneD',
                        choices=["oneD", "single_integrator", "double_integrator", "car", "car_park", "manipulator", "ur5"],
                        help="System-id (single_integrator, double_integrator, car, manipulator, ur5")

    parser.add_argument('--GPU-device',                       type=str,  default="0",
                        help="GPU device to use (set to None to use CPU)")
    
    parser.add_argument('--plot-flag',                        type=bool,  default=True,
                        choices=[True, False],
                        help="Flag to plot results")
    
    parser.add_argument('--w-S',                              type=float, default=0,
                        help="Sobolev training - weight of the value related error")

    parser.add_argument('--BICSf',                            type=float, default=0,
                        help="BICS factor")
    
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
        'oneD': ('conf_oneD', 'OneD', 'OneD_CAMS'),
        'single_integrator': ('conf_single_integrator', 'SingleIntegrator_CAMS'),
        'double_integrator': ('conf_double_integrator', 'DoubleIntegrator_CAMS'),
        'car':               ('conf_car', 'Car_CAMS'),
        'car_park':          ('conf_car_park', 'CarPark_CAMS'),
        'manipulator':       ('conf_manipulator', 'Manipulator_CAMS'),
        'ur5':               ('conf_ur5', 'UR5_CAMS')
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



    ### Create instances of the used classes ###
    env_TO = Environment_TO                         # Create an instance of the environment

    TrOpU = TO_JAX(env_TO, conf)                    # Create an instance of the TO solver for each TO batch size
    if BICS_factor > 0:
        TrOpB = TO_JAX(env_TO, conf)
    TrOpPlot =  TO_JAX(env_TO, conf) 

    buffer = ReplayBuffer_JAX(conf)                 # Create an empty replay buffer
    plot_fun = PLOT(N_try, env_TO, TrOpPlot, conf)  # Create an instance of plot utils                                                                       
    
    ######################################
    from jaxadi import convert
    CAMS = env_TO
    runningSingleModel = CAMS('running_model', conf)
    cost_func_aug = jax.jit(convert(runningSingleModel.cost_aug_tau))
    dynamics_func = jax.jit(convert(runningSingleModel.x_next_aug_tau_so))
    check_feasible = jax.jit(convert(runningSingleModel.check_feasible))
    @jax.jit
    def cost_func_aug_nolist(x, u):
        return jnp.squeeze(cost_func_aug(x, u)[0])
    @jax.jit
    def dynamics_func_nolist(x, u):
        return jnp.squeeze(dynamics_func(x, u)[0])
    @jax.jit
    def check_feasible_nolist(x):
        return jnp.squeeze(check_feasible(x)[0])
    ######################################                                                                    
    
    import flax
    import optax
    from flax.training.train_state import TrainState

    class TrainStateC(TrainState):
        target_params: flax.core.FrozenDict

    @jax.jit
    def update_critic(
            critic_state: TrainState,
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            dV_values: np.ndarray,
            terminations: np.ndarray,
            ):
        ''' Update critic network '''
        
        # Compute target values
        critic_next_target = critic.apply(critic_state.target_params, next_observations).reshape(-1)
        next_critic_value = (rewards.reshape(-1) + (1 - terminations).reshape(-1) * (critic_next_target)).reshape(-1)

        def mse_loss(params):
            # Compute critic values and gradients
            critic_values = critic.apply(params, observations).squeeze()
            dcritic_values = jax.vmap(jax.grad(lambda s: critic.apply(params, s).squeeze()))(observations)
            
            return (w_S*(critic_values - next_critic_value)** 2 + jnp.mean((custom_log(dcritic_values[:,:-1]) - custom_log(dV_values[:,:-1]))**2, axis=1).reshape(-1)).mean() + get_regularization(params, 0, 1e-3)

        # Compute loss and gradients
        critic_loss_value, grads = jax.value_and_grad(mse_loss)(critic_state.params)

        # Update critic network
        critic_state = critic_state.apply_gradients(grads=grads)

        # Update critic target network
        critic_state = critic_state.replace(
            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, conf.UPDATE_RATE)
        )
        
        return critic_state, critic_loss_value
    
    @jax.jit
    def update_std_critic(
            std_critic_state: TrainState,
            critic_state: TrainState,
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            ):
        ''' Update std critic network '''
        
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
            
            return (std_critic_values + critic_error / exp_std_critic_values).mean() + get_regularization(params, 0, 1e-1)

        # Compute loss and gradients
        std_critic_loss_value, grads = jax.value_and_grad(mse_loss)(std_critic_state.params)
        
        # Update std critic network
        std_critic_state = std_critic_state.apply_gradients(grads=grads)

        return std_critic_state, std_critic_loss_value
    
    @jax.jit
    def update_actor(
        actor_state: TrainState,
        critic_state: TrainState,
        observations: np.ndarray,
        ):
        ''' Update actor network '''
        
        # Discrete time update
        def actor_loss(params):
            # Compute actions and the corresponding next states
            tau = actor.apply(params, observations)
            x_next =  jax.vmap(dynamics_func_nolist)(observations, tau)
            return (critic.apply(critic_state.params, x_next).squeeze()/conf.norm_factor + jax.vmap(cost_func_aug_nolist)(observations, tau).squeeze()).mean() + get_regularization(params, 0.01, 0.01)
        
        # Continous time update
        #tau = actor.apply(actor_state.params, observations)
        #dcritic_dstate = jax.vmap(jax.grad(lambda s: critic.apply(critic_state.params, s).squeeze()))(observations)[:, None, :]
        #ddyn_da = jax.vmap(jax.jacrev(dynamics_func_nolist, argnums=1))(observations, tau)/conf.norm_factor #.transpose(1,0,2)
        #dcritic_da = (dcritic_dstate @ ddyn_da/conf.dt*conf.dt).squeeze()[:, None, :]
        #def actor_loss_CT(params):
        #    tau = actor.apply(params, observations)
        #    return ((dcritic_da @ tau[:, :, None]).squeeze() + jax.vmap(cost_func_aug_nolist)(observations, tau).squeeze()).mean() + get_regularization(params, 0.01, 0.01)
    
        # Compute loss and gradients
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)

        # Update actor network
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, critic_state, actor_loss_value


    def create_unif_TO_init(n_UICS, subkey):
        ''' Create n uniformely distributed ICS '''
        # Create uniform ICS
        init_rand_state_t = jax.random.uniform(subkey, shape=(n_UICS, conf.nb_state), minval=conf.x_init_min, maxval=conf.x_init_max)
        init_rand_state = init_rand_state_t[:,:-1]
        init_rand_t = conf.dt*jnp.ceil(init_rand_state_t[:,-1]/conf.dt)
        
        return init_rand_state, init_rand_t
        
    def create_biased_TO_init(n_BICS, std_critic_state, subkey, factor=10):
        ''' Create n biased ICS '''
        # Create biased ICS
        init_rand_state_t = jax.random.uniform(subkey, shape=(n_BICS*factor, conf.nb_state), minval=conf.x_init_min, maxval=conf.x_init_max)

        mask = (jax.vmap(check_feasible_nolist)(init_rand_state_t)) == 1
        feas_init_rand_state_t = init_rand_state_t[mask]

        def eval_batch(x):
            return std_critic.apply(std_critic_state.params, x).squeeze()
        std_values = jnp.exp(jax.vmap(eval_batch)(feas_init_rand_state_t))

        idx = jnp.argsort(std_values.squeeze())
        sorted_init_rand_state_t = feas_init_rand_state_t[idx].squeeze()

        return sorted_init_rand_state_t[-n_BICS:, :-1], conf.dt*jnp.ceil(sorted_init_rand_state_t[-n_BICS:, -1]/conf.dt)



    NSTEPS = conf.NSTEPS
    n_TO_U = conf.EP_UPDATE
    
    start = time.time()
    TrOpU.TO_System_Solve(jnp.zeros((n_TO_U, conf.nx)), jnp.zeros((n_TO_U, NSTEPS, conf.na)), jnp.ones(n_TO_U), maxiter=0)
    
    if BICS_factor > 0:
        n_TO_B = int(n_TO_U*BICS_factor)
        TrOpB.TO_System_Solve(jnp.zeros((n_TO_B, conf.nx)), jnp.zeros((n_TO_B, NSTEPS, conf.na)), jnp.ones(n_TO_B), maxiter=0)
    print('Elapsed time TO init: ', time.time()-start)



    ### START TRAINING ###
    if conf.profile:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()



    time_start_init = time.time()

    # Initialize networks
    actor, critic, std_critic, actor_state, critic_state, std_critic_state = create_networks(conf, seed=seed)  
    save_weights(conf.NNs_path, actor_state.params, N_try, 0)

    # Initialize step counter
    update_step_counter = 0

    # Create random key
    key = jax.random.PRNGKey(seed)

    # Training loop
    for ep in range(conf.NLOOPS): 
        print('#### EPISODE {} ####'.format(ep))
        if plot_flag:
            if ep == 0:
                plot_fun.plot_traj_from_ICS_jax(np.array(conf.init_states_sim), lambda x: actor.apply(actor_state.params, x), update_step_counter=0, steps=conf.NSTEPS, init=0)

        # Create random keys
        key, subkey_create_init, subkey_sample_ac, subkey_sample_stdc = jax.random.split(key, 4)

        # Create initial states for the TO problem
        if ep > 0 and BICS_factor > 0:
            init_rand_state, init_rand_t = create_biased_TO_init(n_TO_B, std_critic_state, subkey_create_init, factor=10)
        else:
            init_rand_state, init_rand_t = create_unif_TO_init(n_TO_U, subkey_create_init)
        NSTEPS_SH = conf.NSTEPS - jnp.floor(init_rand_t / conf.dt)

        if plot_flag:
            plot_fun.plot_ICS(np.array(init_rand_state), name='ICS_{}'.format(ep))

        # Create warmstart for the TO problem
        time_start = time.time()
        if ep > 0:
            TrOp = TrOpB if BICS_factor > 0 else TrOpU
            init_TO_states, init_TO_controls, success_init_flag = TrOp.create_TO_ws(init_rand_state, init_rand_t, lambda x: actor.apply(actor_state.params, x))            
        else:      
            TrOp = TrOpB          
            init_TO_controls = jnp.zeros((n_TO_U, NSTEPS, conf.nb_action))
        print('Create init TO : ', time.time()-time_start)

        # Solve the TO problem
        time_start = time.time()
        TO_states, TO_controls, TO_ee_pos_arr, TO_step_cost, sf, dVdx = TrOp.TO_System_Solve(init_rand_state, init_TO_controls, NSTEPS_SH, maxiter=250, psd_delta=1e-6)
        print('Solve TO       : ', time.time()-time_start)

        # Postprocess TO data
        time_start = time.time()
        TrOp = TrOpB if (ep > 0 and BICS_factor > 0) else TrOpU
        state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ep_return, RL_ee_pos_arr, dVdx_arr  = jax.vmap(TrOp.postprocess_TO_data)(TO_controls, TO_states, TO_step_cost, TO_ee_pos_arr, NSTEPS_SH, dVdx, init_rand_t)
        print('Postprocess TO : ', time.time()-time_start)

        # Remove unvalid data (unsuccess TO or out of the horizon)
        time_start = time.time()
        success_mask = sf != 0
        indices = jnp.arange(conf.NSTEPS+1)
        valid_mask = indices[None, :] < NSTEPS_SH[:, None]
        mask = success_mask[:, None] & valid_mask
        data = [state_arr[mask].reshape(-1,conf.nx+1), partial_reward_to_go_arr[mask].reshape(-1,1), state_next_rollout_arr[mask].reshape(-1,conf.nx+1), dVdx_arr[mask].reshape(-1,conf.nx+1), done_arr[mask].reshape(-1,1)]
        
        # Update the buffer
        buffer.add(data)
        print('Store data     : ', time.time()-time_start)

        # Update actor-critic
        t_c, t_a = 0, 0

        state_batch, partial_reward_to_go_batch, state_next_rollout_batch, dVdx_batch, d_batch, weights_batch, batch_idxes = buffer.sample(subkey_sample_ac, n_sample=int(conf.UPDATE_LOOPS[ep]))
        for i in range(int(conf.UPDATE_LOOPS[ep])):
            t_s = time.time()
            critic_state, critic_loss = update_critic(critic_state, state_batch[i], state_next_rollout_batch[i], partial_reward_to_go_batch[i], dVdx_batch[i], d_batch[i])
            t_c += time.time()-t_s

            t_s = time.time()
            actor_state, critic_state, actor_loss = update_actor(actor_state, critic_state, state_batch[i])
            t_a += time.time()-t_s

            # Save weights
            if update_step_counter % conf.save_interval == 0:
                save_weights(conf.NNs_path, actor_state.params, N_try, update_step_counter)
            
            # Update step counter
            update_step_counter += 1
            if update_step_counter > conf.NUPDATES:
                break

        # Update std critic
        t_stdc = 0 

        state_batch, partial_reward_to_go_batch, state_next_rollout_batch, dVdx_batch, d_batch, weights_batch, batch_idxes = buffer.sample(subkey_sample_stdc, n_sample=int(conf.UPDATE_LOOPS[ep]))
        if BICS_factor > 0:
            for i in range(int(conf.UPDATE_LOOPS[ep])):
                t_s = time.time()
                std_critic_state, std_critic_loss = update_std_critic(std_critic_state, critic_state, state_batch[i], state_next_rollout_batch[i], partial_reward_to_go_batch[i], d_batch[i])
                t_stdc += time.time()-t_s

        print(ep, 'TO problem solved: {}'.format(jnp.sum(success_mask)), 'Critic time: {:3f} '.format(t_c), 'Actor time: {:3f} '.format(t_a), 'STDCritic time: {:3f}'.format(t_stdc))
    
        if plot_flag:
            # plot Critic, Target and STD Critic functions
            if system_id != 'manipulator':
                plot_fun.plot_Critic_Value_function(lambda x: critic.apply(critic_state.params, x), ep, system_id, name='V') ###
                plot_fun.plot_Critic_Value_function(lambda x: critic.apply(critic_state.target_params, x), ep, system_id, name='T')
                plot_fun.plot_Critic_Value_function(lambda x: std_critic.apply(std_critic_state.params, x), ep, system_id, name='S') ###
    
            # Plot rollouts and state and control trajectories
            if update_step_counter%conf.plot_rollout_interval_diff_loc == 0 or 1 < 2:
                print("System: {} - N_try = {}".format(conf.system_id, N_try))
                plot_fun.plot_traj_from_ICS_jax(np.array(conf.init_states_sim), lambda x: actor.apply(actor_state.params, x), update_step_counter=ep, steps=conf.NSTEPS, psd_delta=1e-4)

        if update_step_counter > conf.NUPDATES:
            break

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start_init)

    if conf.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
