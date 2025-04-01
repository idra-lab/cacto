import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse, Rectangle
#from utils import *

import jax
import jax.numpy as jnp

class PLOT():
    def __init__(self, N_try, env_TO, TrOp, conf):
        '''    
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
        '''
        self.conf = conf

        self.N_try = N_try

        self.xlim = conf.fig_ax_lim[0].tolist()
        self.ylim = conf.fig_ax_lim[1].tolist()

        # Set the ticklabel font size globally
        plt.rcParams['xtick.labelsize'] = 22
        plt.rcParams['ytick.labelsize'] = 22
        plt.rcParams.update({'font.size': 20})

        self.p_ee = env_TO('running_model', self.conf).p_ee_jax
        self.TrOp = TrOp

    def plot_obstaces(self, a=1):
        if self.conf.system_id == 'car_park':
            obs1 = Rectangle((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1, 0.0,alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Rectangle((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2, 0.0,alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Rectangle((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3, 0.0,alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])

            #rec1 = FancyBboxPatch((self.conf.XC1-self.conf.A1/2, self.conf.YC1-self.conf.B1/2), self.conf.A1, self.conf.B1,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec1.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec2 = FancyBboxPatch((self.conf.XC2-self.conf.A2/2, self.conf.YC2-self.conf.B2/2), self.conf.A2, self.conf.B2,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec2.set_facecolor([30/255, 130/255, 76/255, 1])
            #rec3 = FancyBboxPatch((self.conf.XC3-self.conf.A3/2, self.conf.YC3-self.conf.B3/2), self.conf.A3, self.conf.B3,edgecolor='g', boxstyle='round,pad=0.1',alpha=a)
            #rec3.set_facecolor([30/255, 130/255, 76/255, 1])

            obs_list = [obs1, obs2, obs3]
        elif self.conf.system_id != 'oneD':
            obs1 = Ellipse((self.conf.XC1, self.conf.YC1), self.conf.A1, self.conf.B1, alpha=a)
            obs1.set_facecolor([30/255, 130/255, 76/255, 1])
            obs2 = Ellipse((self.conf.XC2, self.conf.YC2), self.conf.A2, self.conf.B2, alpha=a)
            obs2.set_facecolor([30/255, 130/255, 76/255, 1])
            obs3 = Ellipse((self.conf.XC3, self.conf.YC3), self.conf.A3, self.conf.B3, alpha=a)
            obs3.set_facecolor([30/255, 130/255, 76/255, 1])
            obs_list = [obs1, obs2, obs3]
        else:
            obs_list = []

        return obs_list
    
    def compute_ICS(self, p_ee, sys_id, theta=None, continue_flag=0):
        if sys_id == 'manipulator':
            radius = math.sqrt((p_ee[0]-self.conf.x_base)**2+(p_ee[1])**2)
            if radius > self.conf.l*3:
                continue_flag = 1
                return None, continue_flag

            phi = math.atan2(p_ee[1]-self.conf.y_base,(p_ee[0]-self.conf.x_base))               # SUM OF THE ANGLES FIXED   
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

        elif sys_id == 'car':
            if theta == None:
                theta = 0*np.random.uniform(-math.pi,math.pi)
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'car_park':
            if theta == None:
                #theta = 0*np.random.uniform(-math.pi,math.pi)
                theta = np.pi/2
            ICS = np.array([p_ee[0], p_ee[1], theta, 0.0, 0.0, 0.0])

        elif sys_id == 'double_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0, 0.0, 0.0])
        
        elif sys_id == 'single_integrator':
            ICS = np.array([p_ee[0], p_ee[1], 0.0])

        elif sys_id == 'oneD':
            ICS = np.array([p_ee[0], 0.0, 0.0])

        elif sys_id == 'bicopter':
            ICS = np.array([p_ee[0], p_ee[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return ICS, continue_flag
    
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
                p_arr[j,:] = self.p_ee(input_arr[j].reshape(-1,1))[0].reshape(-1,)
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

    def plot_traj_from_ICS_jax(self, init_state, actor, update_step_counter=0, steps=200, init=1, psd_delta=1e-6):
        ''' Plot results from TO and episode to check consistency '''
        def get_ee_pos(init_state):
            init_TO_states, init_TO_controls, _ = self.TrOp.create_TO_ws(jnp.array([init_state[:-1]]), jnp.zeros_like(init_state[-1]), actor, init=init)
            _, _, TO_p_ee, _, sf, _  = self.TrOp.TO_System_Solve(jnp.array([init_state[:-1]]), init_TO_controls, (steps)*jnp.ones(1), maxiter=10000, psd_delta=psd_delta)

            ee_pos_RL = jax.vmap(self.TrOp.p_ee_jax)(init_TO_states.squeeze()[:,:,None])[0].squeeze()
            ee_pos_TO = jnp.where(sf == 1, TO_p_ee.squeeze(), 0*TO_p_ee.squeeze())
            
            return ee_pos_RL, ee_pos_TO

        ee_pos_RL, ee_pos_TO = jax.vmap(get_ee_pos)(init_state)  
        ee_pos_RL, ee_pos_TO = np.array(ee_pos_RL), np.array(ee_pos_TO)


        colors = cm.coolwarm(np.linspace(0.1,1,len(init_state)))

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        for j in range(len(init_state)):
            ax1.scatter(ee_pos_RL[j, 0, 0], ee_pos_RL[j, 0, 1], color=colors[j], label=f'Start {j}')
            ax1.plot(ee_pos_RL[j, :, 0], ee_pos_RL[j, :, 1], '--', color=colors[j])
            ax2.scatter(ee_pos_TO[j, 0, 0], ee_pos_TO[j, 0, 1], color=colors[j])
            ax2.plot(ee_pos_TO[j, :, 0], ee_pos_TO[j, :, 1], color=colors[j])
            
        ax1.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
        ax2.plot([self.conf.TARGET_STATE[0]],[self.conf.TARGET_STATE[1]],'b*',markersize=5) 
        
        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax1.add_patch(obs_plot_list[i])

        obs_plot_list = self.plot_obstaces(a=0.5)
        for i in range(len(obs_plot_list)):
            ax2.add_patch(obs_plot_list[i])

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

        plt.savefig('{}/N_try_{}/ee_traj_{}_{}'.format(self.conf.Fig_path,self.N_try,init,update_step_counter))
        print('ICS plot saved')
        plt.close()

    def plot_Critic_Value_function(self, critic_model, n_update, sys_id, name='V'):
        ''' Plot Value function as learned by the critic '''
        N_discretization_x = 30 + 1  
        N_discretization_y = 30 + 1

        plot_data = np.zeros((N_discretization_y,N_discretization_x))*np.nan

        if sys_id == 'maniupulator':
            ee_x = np.linspace(-37, 23, N_discretization_x)
            ee_y = np.linspace(-30, 30, N_discretization_y)
        else:
            ee_x = np.linspace(-15, 15, N_discretization_x)
            ee_y = np.linspace(-15, 15, N_discretization_y)

        for k_y in range(N_discretization_y):
            for k_x in range(N_discretization_x):
                p_ee = np.array([ee_x[k_x], ee_y[k_y], 0])
                ICS, continue_flag = self.compute_ICS(p_ee, sys_id, continue_flag=0)
                if continue_flag:
                    continue
                plot_data[k_x,k_y] = critic_model(jnp.array(ICS))[0]

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