import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Ellipse, Rectangle

import jax
import jax.numpy as jnp

class PLOT():
    def __init__(self, N_try, env_TO, TrOp, conf):
        """   
        :input N_try :                          (Test number)

        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)
            :param fig_ax_lim :                 (float array) Figure axis limit [x_min, x_max, y_min, y_max]
            :param Fig_path :                   (str) Figure path
            :param NSTEPS :                     (int) Max episode length
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param dt :                         (float) Timestep
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
       """
        self.conf = conf

        self.N_try = N_try

        self.xlim = conf.fig_ax_lim[0].tolist()
        self.ylim = conf.fig_ax_lim[1].tolist()

        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22
        plt.rcParams.update({'font.size': 20})

        self.p_ee = env_TO.p_ee_jax_wrapped
        self.cost = env_TO.cost_tau_func_wrapped
        self.dyn = env_TO.dynamics_tau_func_wrapped
        self.TrOp = TrOp

    def plot_obstaces(self, x=None, a=1):
        if self.conf.system_id == 'reacher':
            obs_list = []
            
        elif self.conf.system_id == 'aliengo':
            if x is None:
                obs1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, alpha=a)
                obs1.set_facecolor([30/255, 130/255, 76/255, 1])
                obs2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, alpha=a)
                obs2.set_facecolor([30/255, 130/255, 76/255, 1])
                obs3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, alpha=a)
                obs3.set_facecolor([30/255, 130/255, 76/255, 1])
                obs4 = Ellipse((self.conf.XC4, self.conf.YC4), self.conf.A4, self.conf.B4, alpha=a)
                obs4.set_facecolor([30/255, 130/255, 76/255, 1])
                obs5 = Ellipse((self.conf.XC5, self.conf.YC5), self.conf.A5, self.conf.B5, alpha=a)
                #obs5.set_facecolor([30/255, 130/255, 76/255, 1])
            else:
                obstacle_state = x[8:10]
                wall_state = x[10:14]
                obs1 = Ellipse((                   wall_state[0], (wall_state[1]+wall_state[3])/2),                  self.conf.A1, wall_state[1]-wall_state[3], alpha=a)
                obs2 = Ellipse((((wall_state[0]+wall_state[2])/2,                  wall_state[1])), (wall_state[0]-wall_state[2]),                  self.conf.B2, alpha=a)
                obs3 = Ellipse((                   wall_state[2], (wall_state[1]+wall_state[3])/2),                  self.conf.A3, wall_state[1]-wall_state[3], alpha=a)
                obs4 = Ellipse((((wall_state[0]+wall_state[2])/2,                  wall_state[3])), (wall_state[0]-wall_state[2]),                  self.conf.B4, alpha=a)
                obs5 = Ellipse(((obstacle_state[0], obstacle_state[1])), self.conf.A5, self.conf.B5, alpha=a)
                obs1.set_facecolor([30/255, 130/255, 76/255, 1])
                obs2.set_facecolor([30/255, 130/255, 76/255, 1])
                obs3.set_facecolor([30/255, 130/255, 76/255, 1])
                obs4.set_facecolor([30/255, 130/255, 76/255, 1])
                #obs5.set_facecolor([30/255, 130/255, 76/255, 1])
            obs_list = [obs1, obs2, obs3, obs4, obs5]
        else:
            obs1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])
            obs_list = [obs1, obs2, obs3]

        return obs_list
    
    def plot_body(self, init_state, theta=0, facecolor=None, a=0.1):
        list_body = []
        for i in range(len(init_state)):
            angle = jnp.arctan2(init_state[i,9],init_state[i,8])
            list_body.append(Rectangle((init_state[i,4]-self.conf.lx_tot, init_state[i,5]-self.conf.ly_tot), self.conf.lx_tot*2, self.conf.ly_tot*2, angle=angle*180/np.pi, rotation_point='center', alpha=a))
            if facecolor is not None:
                list_body[i].set_facecolor(facecolor[i])
    
        return list_body
    
    def compute_ICS(self, p_ee, sys_id, theta=None, continue_flag=0):
        if sys_id == 'manipulator':
            radius = math.sqrt((p_ee[0]-self.conf.x_base)**2+(p_ee[1])**2)
            if radius > self.conf.l*3:
                continue_flag = 1
                return None, continue_flag

            phi = math.atan2(p_ee[1]-self.conf.y_base,(p_ee[0]-self.conf.x_base)) 
            X3rd_joint = (p_ee[0]-self.conf.x_base) - self.conf.l* math.cos(phi) 
            Y3rd_joint = (p_ee[1]-self.conf.y_base) - self.conf.l* math.sin(phi)

            if abs(X3rd_joint) <= 1e-6 and abs(Y3rd_joint) <= 1e-6:
                continue_flag = 1
                return None, continue_flag

            c2 = (X3rd_joint**2 + Y3rd_joint**2 -2*self.conf.l**2)/(2*self.conf.l**2)

            if p_ee[1] >= 0:
                s2 = math.sqrt(1-c2**2)
            else:
                s2 = -math.sqrt(1-c2**2)

            s1 = ((self.conf.l + self.conf.l*c2)*Y3rd_joint - self.conf.l*s2*X3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)  
            c1 = ((self.conf.l + self.conf.l*c2)*X3rd_joint - self.conf.l*s2*Y3rd_joint)/(X3rd_joint**2 + Y3rd_joint**2)
            ICS_q0 = math.atan2(s1,c1)
            ICS_q1 = math.atan2(s2,c2)
            ICS_q2 = phi-ICS_q0-ICS_q1

            ICS = np.array([ICS_q0, ICS_q1, ICS_q2, 0.0, 0.0, 0.0, 0.0])

        elif sys_id == 'reacher':
            dx = p_ee[0] - 0
            dy = p_ee[1] - 0
            r = math.sqrt(dx**2 + dy**2)
    
            if r > (0.1+0.11) or r < 1e-6:
                continue_flag = 1
                return None, continue_flag
    
            c1 = (dx**2 + dy**2 - (0.1+0.11)**2) / ((0.1+0.11)**2)
            c1 = max(min(c1, 1.0), -1.0)
            q1_elbow_down = math.acos(c1)
            q1_elbow_up = -math.acos(c1)
    
            def shoulder(q1):
                return math.atan2(dy, dx) - math.atan2(0.11*math.sin(q1), 0.1 + 0.11*math.cos(q1))
    
            q0_elbow_down = shoulder(q1_elbow_down)
            ###q0_elbow_up = shoulder(q1_elbow_up)
    
            ICS = np.array([q0_elbow_down, q1_elbow_down, 0.0, 0.0, self.conf.x_des, self.conf.y_des, self.conf.x_des-dx, self.conf.y_des-dy, 0.0])

        elif sys_id == 'car':
            if theta == None:
                theta = 0*np.random.uniform(-math.pi,math.pi)
            ICS = np.array([p_ee[0], p_ee[1], 1.0, 0.0, 0.0, 0.0, 0.0])

        elif sys_id == 'car_park':
            if theta == None:
                theta = np.pi/2
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'double_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0, 0.0, 0.0])
        
        elif sys_id == 'single_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0])
        
        elif sys_id == 'aliengo':
            ICS = np.array([0.0, 0.0, 0.0, 0.0, p_ee[0], p_ee[1], 0.0, 0.0, self.conf.XC5, self.conf.YC5, self.conf.XC1, self.conf.YC2, self.conf.XC3, self.conf.YC4, ((p_ee[0]-self.conf.XC5)**2+(p_ee[1]-self.conf.YC5)**2)**0.5, ((p_ee[0])**2+(p_ee[1])**2)**0.5, 0.0])

        return ICS, continue_flag
    
    def plot_traj_from_ICS_jax_WSONLY(self, init_state, actor, actor_params, obstacle_state=None, sub_name=0, steps=200, init=1, psd_delta=1e-6, name='ee_traj'):
        ''' Plot results from TO and episode to check consistency '''
        if obstacle_state is not None:
            init_state[:, 8] = obstacle_state[0]
            init_state[:, 9] = obstacle_state[1]

        init_TO_states, init_TO_controls, _ = self.TrOp.create_TO_ws(init_state, actor.apply, actor_params, init=init)
        def get_ee_pos(init_TO_states, init_TO_controls):
            ee_pos_RL = jax.vmap(self.p_ee)(init_TO_states.squeeze()[:,:,None])

            return ee_pos_RL, init_TO_states, init_TO_controls
        
        ee_pos_RL, init_TO_states, init_TO_controls = jax.vmap(get_ee_pos)(init_TO_states, init_TO_controls)  
        ee_pos_RL, init_TO_states, init_TO_controls = np.array(ee_pos_RL), np.array(init_TO_states).squeeze(), np.array(init_TO_controls).squeeze()

        colorss = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig, ax1 = plt.subplots(figsize=(12,8))

        for j in range(len(init_state)):
            ax1.plot(ee_pos_RL[j, :-1, 0], ee_pos_RL[j, :-1, 1], color=colorss[j])
        
        ee_pos_RL_reshaped = jnp.reshape(ee_pos_RL[:,:-1,:], (-1, ee_pos_RL.shape[2]))
        
        ax1.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 

        obs_plot_list = self.plot_obstaces(x=init_state[0],a=0.5)
        for i in range(len(obs_plot_list)):
            ax1.add_patch(obs_plot_list[i])

        
        if self.conf.system_id == 'aliengo':
            body_list = self.plot_body(init_state, facecolor=colorss)
            for i in range(len(body_list)):
                ax1.add_patch(body_list[i])

            for j in range(50):
                if j % 10 == 0:
                    body_list = self.plot_body(init_TO_states[:,j,:], facecolor=colorss)
                    for i in range(len(body_list)):
                        ax1.scatter(ee_pos_RL[i, j, 0], ee_pos_RL[i, j, 1], color=colorss[i])
                        ax1.add_patch(body_list[i])
        
        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('CACTO CoM traj. N=50')
        ax1.grid(True)
        plt.savefig('{}_{}_{}'.format(self.conf.Fig_path,self.N_try,name,init,sub_name))
        plt.close(fig)
        plt.close()
    
    def plot_ICS(self, input_arr, cs=0, name='ICS'):
        if cs == 1:
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()
            for j in range(len(input_arr)):
                p_arr[j,:] = input_arr[j,:]
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/{}'.format(self.conf.Fig_path,self.N_try, name))
            plt.close(fig)
        else:    
            p_arr = np.zeros((len(input_arr),3))
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot()

            for j in range(len(input_arr)):
                p_arr[j,:] = self.p_ee(input_arr[j].reshape(-1,1)).reshape(-1,)
            ax.scatter(p_arr[:,0],p_arr[:,1])
            obs_plot_list = self.plot_obstaces(a = 0.5)
            for i in range(len(obs_plot_list)):
                ax.add_artist(obs_plot_list[i]) 
            ax.set_xlim(self.conf.fig_ax_lim[0].tolist())
            ax.set_ylim(self.conf.fig_ax_lim[1].tolist())
            ax.set_aspect('equal', 'box')
            ax.grid()
            plt.savefig('{}/N_try_{}/{}'.format(self.conf.Fig_path,self.N_try,name))
            plt.close(fig)

    def plot_traj_from_ICS_jax(self, init_state, actor_model, actor_params, obstacle_state=None, sub_name=0, steps=200, init=1, maxiter=1000, psd_delta=1e-6, name='ee_traj'):
        """Plot results from TO and episode to check consistency"""
        if obstacle_state is not None:
            init_state[:, 10] = obstacle_state[0]
            init_state[:, 11] = obstacle_state[1]

        if self.conf.system_id == 'reacher':
            p_ee = jax.vmap(self.p_ee)(init_state)
            init_state = init_state.at[:,6:8].set(p_ee[:,:2]-init_state[:,4:6])
        
        init_TO_states, init_TO_controls, _ = self.TrOp.create_TO_ws(init_state, actor_model, actor_params, init=init)
        TO_states, TO_controls, TO_p_ee, cost_arr, sf, _  = self.TrOp.TO_System_Solve(init_state, init_TO_controls, (steps)*jnp.ones(init_state.shape[0]), maxiter=maxiter, psd_delta=psd_delta)

        #for i in range(len(init_TO_controls)):
        #    COST = jax.vmap(self.cost)(init_TO_states[i,:-1,:], init_TO_controls[i,:,:],init_TO_states[i,:-1,4:6])
        #    print(np.sum(COST))

        def get_ee_pos(init_TO_states, init_TO_controls,TO_states, TO_p_ee, cost_arr, sf):

            ee_pos_RL = jax.vmap(self.p_ee)(init_TO_states.squeeze()[:,:,None])#[0].squeeze()
            ee_pos_TO = jnp.where(sf == 1, TO_p_ee.squeeze(), 0*TO_p_ee.squeeze()-1)
            RL_controls = jax.vmap(lambda x: actor_model(actor_params, x))(TO_states.squeeze())
            RL_state_next = jax.vmap(self.dyn)(TO_states.squeeze(), RL_controls)
            RL_controls_next = jax.vmap(lambda x: actor_model(actor_params, x))(RL_state_next.squeeze())
            RL_state_next_next = jax.vmap(self.dyn)(RL_state_next.squeeze(), RL_controls_next)
            init_TO_states = jnp.where(sf == 1, init_TO_states.squeeze(), 0*init_TO_states.squeeze()-1)
            TO_states = jnp.where(sf == 1, TO_states.squeeze(), 0*TO_states.squeeze()-1)
            init_TO_controls = jnp.where(sf == 1, init_TO_controls, 0*init_TO_controls)
            RL_controls = jnp.where(sf == 1, RL_controls.squeeze(), RL_controls.squeeze())
            RL_state_next = jnp.where(sf == 1, RL_state_next.squeeze(), RL_state_next.squeeze()-1)
            RL_controls_next = jnp.where(sf == 1, RL_controls_next.squeeze(), 0*RL_controls_next.squeeze())
            RL_state_next_next = jnp.where(sf == 1, RL_state_next_next.squeeze(), 0*RL_state_next_next.squeeze()-1)
            cost_arr = jnp.sum(jnp.where(sf == 1, cost_arr.squeeze(), 0*cost_arr.squeeze()))

            return ee_pos_RL, ee_pos_TO, cost_arr, sf, init_TO_states, TO_states, init_TO_controls, RL_controls, RL_state_next, RL_controls_next, RL_state_next_next
        
        ee_pos_RL, ee_pos_TO, cost_arr, sf, init_TO_states, TO_states, init_TO_controls, RL_controls, RL_state_next, RL_controls_next, RL_state_next_next = jax.vmap(get_ee_pos)(init_TO_states, init_TO_controls, TO_states, TO_p_ee, cost_arr, sf)  
        ee_pos_RL, ee_pos_TO, cost_arr, sf, init_TO_states, TO_states, init_TO_controls, RL_controls, RL_state_next, RL_controls_next, RL_state_next_next = np.array(ee_pos_RL), np.array(ee_pos_TO), np.array(cost_arr).squeeze(), np.array(sf).squeeze(), np.array(init_TO_states).squeeze(), np.array(TO_states).squeeze(), np.array(init_TO_controls).squeeze(), np.array(RL_controls).squeeze(), np.array(RL_state_next).squeeze(), np.array(RL_controls_next).squeeze(), np.array(RL_state_next_next).squeeze()

        colorss = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        for j in range(len(init_state)):
            ax1.scatter(ee_pos_RL[j, 0, 0], ee_pos_RL[j, 0, 1], color=colorss[j], label=f'Start {j}')
            ax1.plot(ee_pos_RL[j, :, 0], ee_pos_RL[j, :, 1], '--', color=colorss[j])
            ax2.scatter(ee_pos_TO[j, 0, 0], ee_pos_TO[j, 0, 1], color=colorss[j])
            ax2.plot(ee_pos_TO[j, :, 0], ee_pos_TO[j, :, 1], color=colorss[j])
        
        ax1.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
        ax2.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 

        obs_plot_list = self.plot_obstaces(x=init_state[0],a=0.5)
        for i in range(len(obs_plot_list)):
            ax1.add_patch(obs_plot_list[i])

        obs_plot_list = self.plot_obstaces(x=init_state[0],a=0.5)
        for i in range(len(obs_plot_list)):
            ax2.add_patch(obs_plot_list[i])
        
        if self.conf.system_id == 'aliengo':
            body_list = self.plot_body(init_state, facecolor=colorss)
            for i in range(len(body_list)):
                ax1.add_patch(body_list[i])

            body_list = self.plot_body(TO_states[:,-1,:], facecolor=colorss)
            for i in range(len(body_list)):
                ax2.add_patch(body_list[i])
            body_list = self.plot_body(TO_states[:,0,:], facecolor=colorss)
            for i in range(len(body_list)):
                ax2.add_patch(body_list[i])

        if self.conf.system_id == 'reacher':
            ax1.scatter(jnp.array(self.conf.init_states_sim)[:,4],jnp.array(self.conf.init_states_sim)[:,5])
            ax2.scatter(jnp.array(self.conf.init_states_sim)[:,4],jnp.array(self.conf.init_states_sim)[:,5])
        
        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Warmstart traj.')

        ax2.set_xlim(self.xlim)
        ax2.set_ylim(self.ylim)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X [m]')
        #ax2.set_ylabel('Y [m]')
        ax2.set_title('TO traj.')
        #ax.legend()
        ax1.grid(True)
        ax2.grid(True)

        plt.savefig('{}/N_try_{}/{}_{}_{}'.format(self.conf.Fig_path,self.N_try,name,init,sub_name))
        print('ICS plot saved {}'.format(sum(sf == 1)))
        plt.close(fig)
        plt.close()

    def plot_Critic_Value_function(self, critic_model, critic_params, n_update, sys_id, obstacle_state=None, name='V'):
        """Plot Value function as learned by the critic"""
        N_discretization_x = 30 + 1  
        N_discretization_y = 30 + 1

        plot_data = np.zeros((N_discretization_y,N_discretization_x))*np.nan

        if sys_id == 'maniupulator':
            ee_x = np.linspace(-37, 23, N_discretization_x)
            ee_y = np.linspace(-30, 30, N_discretization_y)
        elif sys_id == 'aliengo':
            ee_x = np.linspace(self.conf.fig_ax_lim[0][0], self.conf.fig_ax_lim[0][1], N_discretization_x)
            ee_y = np.linspace(self.conf.fig_ax_lim[1][0], self.conf.fig_ax_lim[1][1], N_discretization_y)
        elif sys_id == 'reacher':
            ee_x = np.linspace(self.conf.fig_ax_lim[0][0], self.conf.fig_ax_lim[0][1], N_discretization_x)
            ee_y = np.linspace(self.conf.fig_ax_lim[1][0], self.conf.fig_ax_lim[1][1], N_discretization_y)
        else:
            ee_x = np.linspace(-15, 15, N_discretization_x)
            ee_y = np.linspace(-15, 15, N_discretization_y)

        for k_y in range(N_discretization_y):
            for k_x in range(N_discretization_x):
                p_ee = np.array([ee_x[k_x], ee_y[k_y], 0])
                ICS, continue_flag = self.compute_ICS(p_ee, sys_id, continue_flag=0)
                if obstacle_state is not None:
                    ICS[-3] = obstacle_state[0]
                    ICS[-2] = obstacle_state[1]

                if continue_flag:
                    continue

                plot_data[k_x,k_y] = critic_model(critic_params, jnp.array(ICS))[0]

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()

            plt.contourf(ee_x, ee_y, plot_data.T, cmap=cm.coolwarm, antialiased=False, levels=50)

            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()

    def plot_reward_function(self, n_update, sys_id, obstacle_state=None, name='r'):
        """Plot Value function as learned by the critic"""
        N_discretization_x = 30 + 1  
        N_discretization_y = 30 + 1

        plot_data = np.zeros((N_discretization_y,N_discretization_x))*np.nan

        if sys_id == 'maniupulator':
            ee_x = np.linspace(-37, 23, N_discretization_x)
            ee_y = np.linspace(-30, 30, N_discretization_y)
        elif sys_id == 'reacher':
            ee_x = np.linspace(-0.25, 0.25, N_discretization_x)
            ee_y = np.linspace(-0.25, 0.25, N_discretization_y)
        elif sys_id == 'aliengo':
            ee_x = np.linspace(self.conf.fig_ax_lim[0][0], self.conf.fig_ax_lim[0][1], N_discretization_x)
            ee_y = np.linspace(self.conf.fig_ax_lim[1][0], self.conf.fig_ax_lim[1][1], N_discretization_y)
        else:
            ee_x = np.linspace(-15, 15, N_discretization_x)
            ee_y = np.linspace(-15, 15, N_discretization_y)

        for k_y in range(N_discretization_y):
            for k_x in range(N_discretization_x):
                p_ee = np.array([ee_x[k_x], ee_y[k_y], 0])
                ICS, continue_flag = self.compute_ICS(p_ee, sys_id, continue_flag=0)
                if obstacle_state is not None:
                    ICS[-3] = obstacle_state[0]
                    ICS[-2] = obstacle_state[1]

                if continue_flag:
                    continue
                
                plot_data[k_x,k_y] = self.cost(jnp.array(ICS), jnp.zeros(self.conf.nb_action),jnp.array(ICS))

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot()

            plt.contourf(ee_x, ee_y, plot_data.T, cmap=cm.coolwarm, antialiased=False, levels=50)

            obs_plot_list = self.plot_obstaces(a=0.5)
            for i in range(len(obs_plot_list)):
                ax.add_patch(obs_plot_list[i])
            plt.colorbar()
            plt.title('N_try {} - n_update {}'.format(self.N_try, n_update))
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', 'box')
            plt.savefig('{}/N_try_{}/{}_{}'.format(self.conf.Fig_path,self.N_try,name,int(n_update)))
            plt.close()