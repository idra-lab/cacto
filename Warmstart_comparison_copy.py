import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # {'0' -> show all logs, '1' -> filter out info, '2' -> filter out warnings}
import math
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial
from TO import TO_Casadi, TO_JAX 
from typing import Any, Sequence
from jax.nn.initializers import glorot_uniform, zeros
from matplotlib import cm

#from utils import *



plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams.update({'font.size': 22})
    
def parse_args():
    ''' Parse the arguments for CACTO training '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sys-id',                      type=str,   default='di')
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args
    
def compare_cost_dic_mean2(N_try, n_update, DictWS_path, runs=[0,1,2]):
    ''' Compute the mean/median cost of the mean cost over the keys (ICS considered)'''
    mean_of_keys = np.zeros(len(runs))*np.nan
    for i in range(len(runs)):
        try:
            f = open("{}/N_try_{}/CACTO_n_upd_{}.pkl".format(DictWS_path, N_try + runs[i], n_update),"rb")
            tmp_CACTO = pickle.load(f)
            f.close()
            keys_to_avg = [key for key in tmp_CACTO.keys() if any([key[0] == ix for ix in discrtized_x]) and any([key[1] == iy for iy in discrtized_y]) and tmp_CACTO[key] is not None]
            mean_of_keys[i] = np.mean([tmp_CACTO[key] for key in keys_to_avg])
        except:
            continue 

    if len(runs) < 4:
        CACTO_mean = np.mean(mean_of_keys)
        CACTO_std  = np.std(mean_of_keys)

        return CACTO_mean, CACTO_mean - CACTO_std, CACTO_mean + CACTO_std
        
    else:
        CACTO_q1   = np.quantile(mean_of_keys, 0.25)
        CACTO_q2   = np.quantile(mean_of_keys, 0.5)
        CACTO_q3   = np.quantile(mean_of_keys, 0.75)
        
        return CACTO_q2, CACTO_q1, CACTO_q3
    
def compute_ICS(i_x, i_y, sys_id, continue_flag=0):
    if sys_id == 'single_integrator':
        ICS = np.array([i_x, i_y, 0.0])

    elif sys_id == 'double_integrator':
        ICS = np.array([i_x, i_y, 0, 0, 0.0])
    
    elif sys_id == 'car':
        ICS = np.array([i_x, i_y, random.uniform(-math.pi,math.pi), 0, 0.0, 0.0])
    
    elif sys_id == 'car_park':
        theta = 0
        c_pos = np.array([i_x, i_y]) - np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).dot(np.array([conf.L/2,0]))
        ICS = np.array([c_pos[0], c_pos[1], theta, 0.0, 0.0, 0.0])

    elif sys_id == 'manipulator':
        ICS_ee = [i_x, i_y]
        radius = math.sqrt((ICS_ee[0]-conf.x_base)**2+(ICS_ee[1])**2)
        if radius > 30:
            obj_dict_CACTO[i_x,i_y] = 1.0
            continue_flag = 1
            return None, continue_flag

        phi = math.atan2(ICS_ee[1]-conf.y_base,(ICS_ee[0]-conf.x_base))               # SUM OF THE ANGLES FIXED   
        X3rd_joint = (ICS_ee[0]-conf.x_base) - conf.l* math.cos(phi) 
        Y3rd_joint = (ICS_ee[1]-conf.y_base) - conf.l* math.sin(phi)

        if abs(X3rd_joint) <= 1e-6 and abs(Y3rd_joint) <= 1e-6:
            obj_dict_CACTO[i_x,i_y] = 0.0
            continue_flag = 1
            return None, continue_flag

        c2 = (X3rd_joint**2 + Y3rd_joint**2 -2*conf.l**2)/(2*conf.l**2)

        if ICS_ee[1] >= 0:
            s2 = math.sqrt(1-c2**2)
        else:
            s2 = -math.sqrt(1-c2**2)

        s1 = ((conf.l + conf.l*c2)*Y3rd_joint - conf.l*s2*X3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)  
        c1 = ((conf.l + conf.l*c2)*X3rd_joint - conf.l*s2*Y3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)
        ICS_q0 = math.atan2(s1,c1)
        ICS_q1 = math.atan2(s2,c2)
        ICS_q2 = phi-ICS_q0-ICS_q1
        ICS = np.array([ICS_q0, ICS_q1, ICS_q2, 0, 0, 0, 0.0])

    return ICS, continue_flag



if __name__ == "__main__":
    #import tikzplotlib

    random.seed(100)
    args = parse_args()
    
    ###           Input           ###
    sys_id = args['sys_id']
    if sys_id == 'm':
        system_id = 'manipulator'
        folder_id = 'Manipulator'
        import conf_manipulator as conf
        from environment_TO import Manipulator_CAMS as Environment_TO

    elif sys_id == 'di':
        system_id = 'double_integrator'
        folder_id = 'Double Integrator'
        import conf_double_integrator as conf
        from environment_TO import DoubleIntegrator_CAMS as Environment_TO

    elif sys_id == 'si':
        system_id = 'single_integrator'
        folder_id = 'Single Integrator'
        import conf_single_integrator as conf
        from environment_TO import SingleIntegrator_CAMS as Environment_TO

    elif sys_id == 'c':
        system_id = 'car'
        folder_id = 'Car'
        import conf_car as conf
        from environment_TO import Car_CAMS as Environment_TO
        
    elif sys_id == 'cp':
        system_id = 'car_park'
        folder_id = 'Car Park'
        import conf_car_park as conf
        from environment_TO import CarPark_CAMS as Environment_TO
    #################################

    # Create environment instance
    env_TO = Environment_TO
    TrOp = TO_JAX(env_TO, conf)                                                                             # Create TO instance
    TrOpCa = TO_Casadi(conf, env_TO)                                                                         # Create TO instance
    
    from jaxadi import convert
    CAMS = env_TO
    runningSingleModel = CAMS('running_model', conf)
    p_ee_func = jax.jit(convert(runningSingleModel.p_ee))
    cost_func_aug = jax.jit(convert(runningSingleModel.cost_aug_tau))
    dynamics_func_tau = jax.jit(convert(runningSingleModel.x_next_aug_tau))
    dynamics_func = jax.jit(convert(runningSingleModel.x_next_aug_tau))
    @jax.jit
    def dynamics_tau_func_nolist(x, u):
        tmp = dynamics_func_tau(x, u)
        return jnp.squeeze(tmp[0]), jnp.squeeze(tmp[1])
    @jax.jit
    def cost_func_aug_nolist(x, u):
        return jnp.squeeze(cost_func_aug(x, u)[0])
    @jax.jit
    def get_end_effector_position(x):
        return jnp.squeeze(p_ee_func(x)[0])

    class ActorMLP(nn.Module):
        features: Sequence[int]

        @nn.compact
        def __call__(self, x):
            if conf.NORMALIZE_INPUTS:
                x = (x + conf.shift) / conf.state_norm_arr
            if conf.system_id == 'manipulator' and conf.remap_angle:
                if x.ndim > 1:
                    cos_angles = jnp.cos(x[:, :3])
                    sin_angles = jnp.sin(x[:, :3])
                    x = jnp.concatenate([cos_angles[:,0].reshape(-1,1), sin_angles[:,0].reshape(-1,1), cos_angles[:,1].reshape(-1,1), sin_angles[:,1].reshape(-1,1), cos_angles[:,2].reshape(-1,1), sin_angles[:,2].reshape(-1,1), x[:, 3:]], axis=1)
                else:
                    x = jnp.hstack([
                        jnp.cos(x[0]), jnp.sin(x[0]), 
                        jnp.cos(x[1]), jnp.sin(x[1]),
                        jnp.cos(x[2]), jnp.sin(x[2]), 
                        x[3], x[4], x[5], x[6]
                    ])
            for feat in self.features[:-1]:
                x = nn.Dense(feat, kernel_init=glorot_uniform(), bias_init=zeros)(x)
                x = nn.leaky_relu(x, negative_slope=0.3)
            x = nn.Dense(self.features[-1], kernel_init=glorot_uniform(), bias_init=zeros)(x)
            x = nn.tanh(x)*conf.u_max
            return x

    from flax.training.train_state import TrainState
    import flax
    import optax
    class TrainStateC(TrainState):
        target_params: flax.core.FrozenDict

    # Create Networks
    def create_actor_network(in_dim, out_dim, layer_sizes_a, learning_rate_a=1, seed=0):
        key1, key_actor, key_critic, key_std_critic = jax.random.split(jax.random.PRNGKey(seed), 4)
        x_dummy = jax.random.normal(key1, (in_dim,))

        # Initialize models
        actor = ActorMLP(layer_sizes_a + [out_dim])

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(key_actor, x_dummy),
            tx=optax.adam(learning_rate=learning_rate_a),
            )
        
        actor.apply = jax.jit(actor.apply)

        return actor, actor_state

    def load_weights(n_try, update_step_counter='final'):
        ''' Save NN weights '''
        with open(conf.NNs_path+"/N_try_{}/actor_{}.pkl".format(n_try,update_step_counter), 'rb') as f:
            actor_params = pickle.load(f)
        return actor_params

    @partial(jax.jit, static_argnames=("dynamics_func", "actor_model"))
    def create_TO_init(ICS, t_ICS, actor_model, dynamics_func):
        ''' Create initial state and initial controls for TO using JAX ''' 
       
        def single_rollout(x0):
            """Runs a single trajectory using lax.scan"""
            def dynamics_for_scan(x, _):
                tau = actor_model(x)
                tmp = dynamics_func(x, tau.reshape(-1, 1))
                x_next = jnp.squeeze(tmp[0])
                u = jnp.squeeze(tmp[1])

                return x_next, (x_next, u) 
            
            _, (X, U) = jax.lax.scan(dynamics_for_scan, x0, jnp.arange(conf.NSTEPS))
            X = jnp.vstack((x0, X))

            return X, U
        
        x0 = jnp.concatenate((ICS, jnp.zeros((ICS.shape[0],1))*conf.dt), axis=1)
        X_batch, U_batch = jax.vmap(single_rollout)(x0)

        success_init_flag = 1

        return X_batch, U_batch, success_init_flag
    
    @jax.jit
    def check_ICS_feasible(state):
        ''' Check if ICS is feasible using JAX '''
        p_ee = get_end_effector_position(state.reshape(-1,1))
        
        ellipses = jnp.array([
            ((p_ee[0] - conf.XC1) ** 2) / ((conf.A1 / 2) ** 2) + ((p_ee[1] - conf.YC1) ** 2) / ((conf.B1 / 2) ** 2),
            ((p_ee[0] - conf.XC2) ** 2) / ((conf.A2 / 2) ** 2) + ((p_ee[1] - conf.YC2) ** 2) / ((conf.B2 / 2) ** 2),
            ((p_ee[0] - conf.XC3) ** 2) / ((conf.A3 / 2) ** 2) + ((p_ee[1] - conf.YC3) ** 2) / ((conf.B3 / 2) ** 2),
        ])

        feasible_flags = jnp.all(ellipses > 1)

        return feasible_flags          

    actor, actor_state = create_actor_network(conf.nb_state, conf.nb_action, [256, 256])

    xlim = conf.fig_ax_lim[0].tolist()
    ylim = conf.fig_ax_lim[1].tolist()

    

    ### Parameters ###
    if system_id == 'car_park':
        N_discretization_x = 20 + 1  
        N_discretization_y = 3 + 1
        discrtized_x = np.linspace(-10, 10, N_discretization_x)
        discrtized_y = np.linspace(1.5, 3, N_discretization_y)
    elif system_id == 'manipulator':
        N_discretization_x = 15 + 1  
        N_discretization_y = 15 + 1
        discrtized_x = np.linspace(-7, 23, N_discretization_x)
        discrtized_y = np.linspace(-15, 15, N_discretization_y)
        #discrtized_x = np.linspace(-37, 23, N_discretization_x)
        #discrtized_y = np.linspace(-30, 30, N_discretization_y)
    else:
        N_discretization_x = 15 + 1  
        N_discretization_y = 10 + 1
        discrtized_x = np.linspace(0, 15, N_discretization_x)
        discrtized_y = np.linspace(-5, 5, N_discretization_y)

    res_id = 'v1_2'
    res_set = 'Results set test - test C' #'Results set test no mix low LrA - 5 runs' #
    res_set_paper = 'Results set test - test C'
    main_path = './Results {}/{}'.format(folder_id,res_set)
    main_path_paper = './Results {}/{}'.format(folder_id,res_set_paper)
    NNs_path_rec = main_path + '/NNs'
    DictWS_path = main_path + '/DictWS_{}'.format(res_id)
    DictWS_path_paper = main_path_paper + '/DictWS_{}'.format(res_id)
    DictWS_path_list = [DictWS_path_paper, DictWS_path]
    legend_list = ['CACTO', 'CACTO-SL']
    if not os.path.exists(main_path):
        print('Chosen directory does not exist')
        import sys
        sys.exit()

    if system_id == 'single_integrator':
        l_bound = 0
        u_bound = 25001
        space   = 5000
    elif system_id == 'double_integrator':
        l_bound = 0
        u_bound = 30001
        space   = 5000
    elif system_id == 'car':
        l_bound = 0
        u_bound = 200001
        space   = 20000
    elif system_id == 'car_park':
        l_bound = 0
        u_bound = 110001
        space   = 10000
    elif system_id == 'manipulator':
        l_bound = 0
        u_bound = 300001
        space   = 30000
    else:
        print('Choose l_bound, u_bound and space')
        import sys
        sys.exit()


    ### Compute parameters ###
    n_try_list_compute = np.array([231,232,233,234,235]) #,141,142,143,144,145,151,152,153,154,155]) + 30#, 3011,3012,3013,3014,3015, 4011,4012,4013,4014,4015, 5011,5012,5013,5014,5015, 6011,6012,6013,6014,6015])
    n_up_list = range(l_bound, u_bound, space)
    n_up_list_rev = range(u_bound-1, l_bound-1, -space)

    
    
    ### Plot parameters ###
    n_try_list_plot = [221,211,231]
    list_weights_wd = n_try_list_plot
    runs = list(range(5))
    colors_list = ['r'] + cm.YlGnBu(np.linspace(0.5,1,len(n_try_list_plot)-1)).tolist()



    compute = 1
    if compute:
        obj_dict_CACTO = {}
        obj_dict_CACTO['total'] = 0

        # Create ICS and remove the unfeasible ones
        ICS = np.zeros((len(discrtized_x)*len(discrtized_y), conf.nb_state))
        i_x_feas = []
        i_y_feas = []
        mask_ICS = np.ones(len(discrtized_x)*len(discrtized_y))
        for idx_x in range(len(discrtized_x)):
            i_x = discrtized_x[idx_x]
            for idx_y in range(len(discrtized_y)):
                i_y = discrtized_y[idx_y]
                obj_dict_CACTO['total'] += 1
                ICS[idx_x*len(discrtized_y)+idx_y], continue_flag = compute_ICS(i_x,i_y,system_id)

                if not check_ICS_feasible(ICS[idx_x*len(discrtized_y)+idx_y]):
                    obj_dict_CACTO[i_x,i_y] = 1.0
                    mask_ICS[idx_x*len(discrtized_y)+idx_y] = 0
                    continue
                                
                i_x_feas.append(i_x)
                i_y_feas.append(i_y)
                        
        # 
        ICS = ICS[mask_ICS == 1, :-1]
        NSTEPS_SH = conf.NSTEPS - jnp.floor(ICS[:,-1] / conf.dt)

        for n_up in n_up_list:
            print(' N_update: ', n_up)
            r = 0
            for n_try in n_try_list_compute:
                r += 1
                # Try to make folder
                try:
                    os.makedirs(DictWS_path + '/N_try_{}'.format(n_try))                                                  
                except:
                    pass

                if not os.path.isfile("{}/N_try_{}/CACTO_n_upd_{}.pkl".format(DictWS_path,n_try,n_up)):
                    #try:
                        
                        obj_dict_CACTO['noconv'] = 0
                        obj_dict_CACTO['failed'] = 0

                        actor_state = actor_state.replace(params=load_weights(n_try, n_up))
                        
                        use_GPU_TO = 1
                        if use_GPU_TO:
                            if n_up == 0:
                                init_TO_controls = jnp.zeros((len(NSTEPS_SH), conf.NSTEPS, conf.nb_action))
                            else:
                                _, init_TO_controls, _ = create_TO_init(ICS, 0*ICS[-1], lambda x: actor.apply(actor_state.params, x), dynamics_func_tau)
                            _, _, _, TO_cost, sf, _ = TrOp.TO_System_Solve(ICS, init_TO_controls, NSTEPS_SH, maxiter=10000, psd_delta=1e-2)
                        else:
                            init_rand_state = jnp.concatenate((ICS, jnp.zeros((len(ICS),1))), axis=1)
                            if n_up == 0:
                                init_TO_controls = jnp.zeros((len(NSTEPS_SH), conf.NSTEPS, conf.nb_action))
                                init_TO_states = jnp.ones((len(NSTEPS_SH), conf.NSTEPS+1, conf.nb_state))*init_rand_state[:, None, :]
                            else:
                                init_TO_states, init_TO_controls, _ = create_TO_init(ICS, 0*ICS[-1], lambda x: actor.apply(actor_state.params, x), dynamics_func_tau)
                            n_TO = len(ICS)

                            TO_states, TO_controls, TO_ee_pos_arr, TO_cost, sf, dVdx = [], [], [], [], [], []
                            for j in range(len(init_rand_state)):
                                TO_states_j, TO_controls_j, TO_ee_pos_arr_j, TO_step_cost_j, sf_j, dVdx_j = TrOpCa.TO_Solve(init_rand_state[j], init_TO_states[j], init_TO_controls[j], int(conf.NSTEPS))
                                TO_states.append(TO_states_j)
                                TO_controls.append(TO_controls_j)
                                TO_ee_pos_arr.append(TO_ee_pos_arr_j)
                                TO_cost.append(TO_step_cost_j)
                                sf.append(sf_j)
                                dVdx.append(dVdx_j)
                            TO_states = jnp.array(TO_states)
                            TO_controls = jnp.array(TO_controls)
                            TO_ee_pos_arr = jnp.array(TO_ee_pos_arr)
                            TO_cost = jnp.array(TO_cost)
                            sf = jnp.array(sf)
                            dVdx = jnp.array(dVdx)

                        for i in range(len(ICS)):
                            i_x = i_x_feas[i]
                            i_y = i_y_feas[i]
                            if sf[i]:
                                obj_dict_CACTO[i_x,i_y] = jnp.sum(TO_cost[i])
                            else:
                                obj_dict_CACTO[i_x,i_y] = 0.0
                                obj_dict_CACTO['failed'] += 1
                                continue

                        print('***** {} *****'.format(n_try))
                        print('Non conv: ', obj_dict_CACTO['noconv'])
                        print('Failed: ', obj_dict_CACTO['failed'])
                        print('Total', obj_dict_CACTO['total'])
                        print('*************')

                        f = open("{}/N_try_{}/CACTO_n_upd_{}.pkl".format(DictWS_path, n_try, n_up),"wb")
                        pickle.dump(obj_dict_CACTO,f)
                        f.close() 

                    #except:
                    #    continue

            #runs = [0,1,2,3,4]

            mean_success_rate_arrayC = np.zeros((len(n_try_list_plot),len(n_up_list)))*np.nan
            low_success_rate_arrayC = np.zeros((len(n_try_list_plot),len(n_up_list)))*np.nan
            high_success_rate_arrayC = np.zeros((len(n_try_list_plot),len(n_up_list)))*np.nan
        
            ### PLOT - QMC ###
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(1,1,1)
            ax.grid(True) 
            succ_rate_c1_arr = np.zeros(len(n_up_list))*np.nan
            for n_up_i in range(len(n_up_list)):
                for n_try_i in range(len(n_try_list_plot)):
                    #try:
                        mean_success_rate_arrayC[n_try_i, n_up_i], low_success_rate_arrayC[n_try_i, n_up_i], high_success_rate_arrayC[n_try_i, n_up_i] = compare_cost_dic_mean2(n_try_list_plot[n_try_i], n_up_list[n_up_i], DictWS_path, runs=runs)#, DictWS_path_compare.index(N_try_comp))
                    #except:
                    #    pass
            for i in range(len(n_try_list_plot)):
                #try:
                    ax.plot(space*np.array(range(len(n_up_list))), mean_success_rate_arrayC[i,:], '-o', label = '{}'.format(list_weights_wd[i]), color=colors_list[i]) #
                    ax.fill_between(space*np.array(range(len(n_up_list))),low_success_rate_arrayC[i,:], high_success_rate_arrayC[i,:], color=colors_list[i], alpha=0.2) 
                #except:
                #    pass
        
            ax.ticklabel_format(scilimits=(0,1))
            ax.set_xlim([0, u_bound])
            ax.legend()
            plt.xlabel('# update')
            plt.ylabel('Mean cost S vs NS')
            plt.savefig('QMC {} {} {} {} {}.png'.format(system_id, res_set, res_id, n_try_list_plot[0], n_try_list_plot[1]))
            plt.close()

    #runs = list(range(len(n_try_list_compute)))

    mean_success_rate_arrayC = np.zeros((len(n_try_list_plot),len(n_up_list)))*np.nan
    low_success_rate_arrayC = np.zeros((len(n_try_list_plot),len(n_up_list)))*np.nan
    high_success_rate_arrayC = np.zeros((len(n_try_list_plot),len(n_up_list)))*np.nan

    x_ep = [[0, 3125*1, 3125*2, 3125*3, 3125*4, 3125*5, 3125*6, 3125*7, 3125*8],
            [0, 2000, 3000, 3500, 4000, 5000, 5500, 6000, 7000]]#, 3000, 3000, 3500, 3500, 3500]]
    ### PLOT - QMC ###
    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,1,1)
    ax.grid(True) 
    succ_rate_c1_arr = np.zeros(len(n_up_list))*np.nan
    for n_up_i in range(len(n_up_list)):
        for n_try_i in range(len(n_try_list_plot)):
            #try:
                mean_success_rate_arrayC[n_try_i, n_up_i], low_success_rate_arrayC[n_try_i, n_up_i], high_success_rate_arrayC[n_try_i, n_up_i] = compare_cost_dic_mean2(n_try_list_plot[n_try_i], n_up_list[n_up_i], DictWS_path_list[n_try_i], runs=runs)#, DictWS_path_compare.index(N_try_comp))
            #except:
            #    pass
    for i in range(len(n_try_list_plot)):
        #try:
            #ax.plot(space*np.array(range(len(n_up_list))), mean_success_rate_arrayC[i,:], '-o', color=colors_list[i], label=legend_list[i]) #
            #ax.fill_between(space*np.array(range(len(n_up_list))),low_success_rate_arrayC[i,:], high_success_rate_arrayC[i,:], color=colors_list[i], alpha=0.2) 
            ax.plot(x_ep[i], mean_success_rate_arrayC[i,:], '-o', color=colors_list[i], label=legend_list[i]) #
            ax.fill_between(x_ep[i],low_success_rate_arrayC[i,:], high_success_rate_arrayC[i,:], color=colors_list[i], alpha=0.2) 
        #except:
        #    pass

    ax.ticklabel_format(scilimits=(0,1))
    ax.set_xlim([0, 25000])
    plt.legend()
    plt.xlabel('# TO episodes')
    plt.ylabel('Mean cost')
    plt.tight_layout() 
    tikzplotlib.save('Q_C.txt')
    #plt.savefig('QMC {} {} {} {} {}.png'.format(system_id, res_set, res_id, n_try_list_plot[0], n_try_list_plot[1]))
    plt.close()



# identifica set e spiega che strategia hai utilizzato