import os
import sys
import time
import shutil
import random
import argparse
import importlib
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import tensorflow as tf
import matplotlib.pyplot as plt
from multiprocessing import Pool
from RL import RL_AC 
from TO import TO_Casadi 
from plot_utils import PLOT
from NeuralNetwork import NN
from replay_buffer import ReplayBuffer

def parse_args():
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test-n',                         type=int,   default=0,                                    
                        help="Test number")
    
    parser.add_argument('--seed',                           type=int,   default=0,                                    
                        help="random and tf.random seed")

    parser.add_argument('--system-id',                      type=str,   default='oneD',
                        choices=["single_integrator", "double_integrator", "car", "manipulator", ],
                        help="System-id (single_integrator, double_integrator, car, manipulator")

    parser.add_argument('--recover-training-flag',          type=bool,  default=False,
                        choices=["True", "False"],
                        help="Flag to recover training")

    parser.add_argument('--GPU-flag',                       type=bool,  default=False,
                        choices=["True", "False"],
                        help="Flag to use GPU")
    
    parser.add_argument('--nb-cpus',                        type=int,   default=10,
                        help="Number of TO problems solved in parallel")
    
    parser.add_argument('--w-S',                            type=float, default=0,
                        help="Sobolev training - weight of the value related error (0 to disable Sobolev training)")

    parser.add_argument('--BICSf',                            type=float, default=0.25,
                        help="BICS factor")
    
    parser.add_argument('--plot-flag',                        type=int,  default=0,
                        choices=[1, 0],
                        help="Flag to plot results")
    
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args




if __name__ == '__main__':

    args = parse_args()
    
    ###############           Input           ###############
    N_try     = args['test_n']

    if args['seed'] == None:
        seed = random.randint(1,100000)
    else:
        seed = args['seed']
    tf.random.set_seed(seed)  # Set tensorflow seed
    random.seed(seed)         # Set random seed

    system_id = args['system_id'] 

    recover_training_flag = args['recover_training_flag']
    
    GPU_flag = args['GPU_flag'] 
    if GPU_flag:
        print(tf.config.experimental.list_physical_devices('GPU'))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

    nb_cpus = args['nb_cpus']

    w_S = args['w_S']

    BICS_factor = args['BICSf']

    plot_flag = args['plot_flag']
    #########################################################



    # Import configuration file and environment file
    system_map = {
        'single_integrator': ('conf_single_integrator', 'SingleIntegrator', 'SingleIntegrator_CAMS'),
        'double_integrator': ('conf_double_integrator', 'DoubleIntegrator', 'DoubleIntegrator_CAMS'),
        'car':               ('conf_car', 'Car', 'Car_CAMS'),
        'manipulator':       ('conf_manipulator', 'Manipulator', 'Manipulator_CAMS'),
    }
    try:
        conf_module, env_class, env_TO_class = system_map[system_id]
        conf = importlib.import_module(conf_module)
        Environment = getattr(importlib.import_module('environment'), env_class)
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

    shutil.copy('{}.py'.format(conf_module), conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try))
    with open(conf.Config_path + '/' + conf_module + '_{}.py'.format(N_try), 'a') as f:
        f.write('\n\n# {}'.format(args))

    # Copy all file with .py extension from /mydir to /mydestdir
    for file in os.listdir("./"):
        if file.endswith(".py"):
            shutil.copy(os.path.join("./", file), os.path.join(conf.Code_path + '/N_try_{}'.format(N_try), file))

    # Create empty txt file in Log_path to store the test info
    open(conf.Log_path + '/info.txt', 'a').close()



    ### Create instances of the used classes ###
    env = Environment(conf)                                                                                 # Create environment instances
    env_TO = Environment_TO
    NN_inst = NN(env, conf, w_S)                                                                            # Create NN instance
    TrOp = TO_Casadi(env, conf, env_TO, w_S)                                                                # Create TO instance
    RLAC = RL_AC(env, NN_inst, conf, N_try)                                                                 # Create RL instance
    buffer = ReplayBuffer(conf)                                                                             # Create an empty (prioritized) replay buffer
    plot_fun = PLOT(N_try, env, env_TO, NN_inst, conf)                                                      # Create PLOT instance

    # Set initial weights of the NNs, initialize the counter of the updates and setup NN models
    if recover_training_flag:
        recover_training = np.array([conf.NNs_path_rec, conf.N_try_rec, conf.update_step_counter_rec])
        update_step_counter = conf.update_step_counter_rec

        RLAC.setup_model(recover_training)
    else:
        update_step_counter = 0

        RLAC.setup_model()

    # Save initial weights of the NNs
    RLAC.RL_save_weights(update_step_counter)

    def compute_sample(args):
        ''' Create samples solving TO problems starting from given ICS '''
        ep = args[0]
        ICS = args[1]

        # Create initial TO #
        init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH, success_init_flag = RLAC.create_TO_init(TrOp, ep, ICS)
        if success_init_flag == 0:
            return None

        # Solve TO problem #
        TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, dVdx = TrOp.TO_Solve(init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
        if success_flag == 0:
            return None
                
        # Collect experiences #
        state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr  = RLAC.RL_Solve(TO_controls, TO_states, TO_step_cost)
        if conf.env_RL == 0:
            RL_ee_pos_arr = TO_ee_pos_arr

        return NSTEPS_SH, TO_controls, TO_ee_pos_arr, dVdx, state_arr.tolist(), partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr

    def create_unif_TO_init(n_UICS=1):
        ''' Create n uniformely distributed ICS '''
        # Create ICS TO #
        init_rand_state = env.reset()
        
        return init_rand_state

    def create_biased_TO_init(n_BICS=1):
        ''' Create n uniformely distributed ICS '''
        # Create ICS TO #
        init_rand_state = env.reset_biased(n_BICS, 10, NN_inst, RLAC)
        
        return init_rand_state
    


    ### START TRAINING ###
    if conf.profile:
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()

    time_start = time.time()
    for ep in range(conf.NLOOPS): 
    
        # Generate and store conf.EP_UPDATE random-uniform ICS
        if ep > 0 and BICS_factor > 0:
            print('BICS')
            EP_UPDATE = int(BICS_factor*conf.EP_UPDATE)
            init_rand_state, std_values, init_rand_state_tmp = create_biased_TO_init(EP_UPDATE)

        else:
            init_rand_state_tmp = None
            EP_UPDATE = conf.EP_UPDATE
            with Pool(nb_cpus) as p: 
                init_rand_state = p.map(create_unif_TO_init, range(EP_UPDATE))

        if plot_flag:
            plot_fun.plot_ICS(init_rand_state, name='ICS_{}'.format(ep))

        # Generate samples
        t_s = time.time()
        with Pool(nb_cpus) as p: 
            tmp = p.map(compute_sample, zip(ep*np.ones(EP_UPDATE), init_rand_state))
        print("Time TO + prepare data: ", time.time()-t_s)

        # Remove unsuccessful TO problems and update EP_UPDATE
        t_s = time.time()
        tmp = [x for x in tmp if x is not None]
        NSTEPS_SH, TO_controls, ee_pos_arr_TO, dVdx, state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL = zip(*tmp)
        
        data = np.concatenate(( np.concatenate(state_arr, axis=0), np.concatenate(partial_reward_to_go_arr, axis=0).reshape(-1,1), np.concatenate(state_next_rollout_arr, axis=0),
                                np.concatenate(dVdx, axis=0), np.concatenate(done_arr, axis=0).reshape(-1,1), np.concatenate(term_arr, axis=0).reshape(-1,1)), axis=1)
        
        # Update the buffer
        buffer.add(data)
        print('Time post process + buffer: ', time.time()-t_s)

        # Update NNs
        t_s = time.time()
        update_step_counter = RLAC.learn_and_update(update_step_counter, buffer, ep, BICS_factor, plot_fun=plot_fun)
        print("Time NN update: ", time.time()-t_s)
        
        # plot Critic value function
        #if plot_flag:
        #    plot_fun.plot_Critic_Value_function(RLAC.std_critic_model, update_step_counter, system_id, name='V')

        # Plot rollouts and state and control trajectories
        if plot_flag:
            print("System: {} - N_try = {}".format(conf.system_id, N_try))
            plot_fun.plot_traj_from_ICS(np.array(conf.init_states_sim), TrOp, RLAC, update_step_counter=update_step_counter, ep=ep,steps=conf.NSTEPS, init=1,NN_inst=NN_inst, ICS=init_rand_state)

        for i in range(len(tmp)):
            print("Episode  {}  --->   Return = {}".format(ep*len(tmp) + i, ep_return[i]))

        if update_step_counter > conf.NUPDATES:
            break

    time_end = time.time()
    print('Elapsed time: ', time_end-time_start)

    if conf.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

    # Save networks at the end of the training
    RLAC.RL_save_weights()

    # Simulate the final policy
    plot_fun.rollout(update_step_counter, RLAC.actor_model, conf.init_states_sim)
