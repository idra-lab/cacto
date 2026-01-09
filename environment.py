import math
import random
import numpy as np
import tensorflow as tf
import pinocchio as pin

from utils import *

class Env:
    def __init__(self, conf):
        '''    
        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param simu :                       (RobotSimulator instance)
            :param x_init_min :                 (float array) State lower bound initial configuration array
            :param x_init_max :                 (float array) State upper bound initial configuration array
            :param x_min :                      (float array) State lower bound vector
            :param x_max :                      (float array) State upper bound vector
            :param u_min :                      (float array) Action lower bound array
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
    '''
        
        self.conf = conf

        self.nq = conf.nq
        self.nv = conf.nv
        self.nx = conf.nx
        self.nu = conf.na

    def reset(self):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(self.conf.nb_state)

        time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1])
        for i in range(self.conf.nb_state-1): 
            state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 
        state[-1] = self.conf.dt*round(time/self.conf.dt)

        return state
    
    def reset_biased(self, n_BICS, factor, NN_inst, RLAC):
        ''' Choose initial state uniformly at random '''
        state_tmp = np.zeros((int(n_BICS*factor),self.conf.nb_state))
        for j in range(int(n_BICS*factor)):
            state_tmp[j,:] = self.reset()

        state_tmp = np.squeeze(np.array([state_tmp[i,:] for i in range(int(n_BICS*factor)) if self.check_ICS_feasible(state_tmp[i,:])]))

        std_values_C = tf.exp(NN_inst.eval(RLAC.std_critic_model, state_tmp)).numpy()

        sorted_indices = np.argsort((std_values_C).flatten())
        
        state = state_tmp[sorted_indices,:]
        state = state[-n_BICS:,:]

        return state, std_values_C, state_tmp

    def check_ICS_feasible(self, state):
        ''' Check if ICS is not feasible '''
        # check if ee is in the obstacles
        p_ee = self.get_end_effector_position(state)

        ellipse1 = ((p_ee[0] - self.conf.XC1)**2) / ((self.conf.A1 / 2)**2) + ((p_ee[1] - self.conf.YC1)**2) / ((self.conf.B1 / 2)**2)
        ellipse2 = ((p_ee[0] - self.conf.XC2)**2) / ((self.conf.A2 / 2)**2) + ((p_ee[1] - self.conf.YC2)**2) / ((self.conf.B2 / 2)**2)
        ellipse3 = ((p_ee[0] - self.conf.XC3)**2) / ((self.conf.A3 / 2)**2) + ((p_ee[1] - self.conf.YC3)**2) / ((self.conf.B3 / 2)**2)
        
        feasible_flag = ellipse1 > 1 and ellipse2 > 1 and ellipse3 > 1

        return feasible_flag
    
    def step(self, weights, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)

        # compute reward
        reward = self.reward(weights, state, action)

        return (state_next, reward)

    def simulate(self, state, action, dt=None):
        ''' Simulate dynamics '''
        if dt is None:
            dt = self.conf.dt

        state_next = np.zeros(self.nx+1)

        # Simulate control action
        self.conf.simu.simulate(np.copy(state[:-1]), action, dt, 1) ### Explicit Euler ###

        # Return next state
        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + dt
        
        return state_next
    
    def derivative(self, state, action, dt=None, CT=0):
        ''' Compute the derivative '''
        if dt is None:
            dt = self.conf.dt
        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]

        # Dynamics gradient w.r.t control (1st order euler)
        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, np.copy(q_init), np.copy(v_init), action)       

        Fu = np.zeros((self.nx+1, self.nu))
        Fu[self.nv:-1, :] = self.conf.robot.data.Minv

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  

        if CT:
            return Fu

        Fu[:self.nx, :] *= dt

        return Fu
    
    def augmented_derivative(self, state, action):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        q = state[:self.nq]
        v = state[self.nq:self.nx]
                
        # Compute Jacobians for continuous time dynamics
        Fx = np.zeros((self.conf.nb_state-1,self.conf.nb_state-1))
        Fu = np.zeros((self.conf.nb_state-1,self.conf.nb_action))

        #pin.computeAllTerms(self.conf.robot.model, self.conf.robot.data, q, v)        
        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, q, v, action)

        Fx[:self.nv, :self.nv] = 0.0
        Fx[:self.nv, self.nv:self.nx] = np.identity(self.nv)
        Fx[self.nv:self.nx, :self.nv] = self.conf.robot.data.ddq_dq
        Fx[self.nv:self.nx, self.nv:self.nx] = self.conf.robot.data.ddq_dv
        Fu[self.nv:self.nx, :] = self.conf.robot.data.Minv
        
        # Convert them to discrete time
        Fx = np.identity(self.conf.nb_state-1) + self.conf.dt * Fx
        Fu *= self.conf.dt
        
        return Fx, Fu

    def simulate_batch(self, state, action, dt=None):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        state_next = np.array([self.simulate(s, a, dt=dt) for s, a in zip(state, action)])

        return tf.convert_to_tensor(state_next, dtype=tf.float32)
        
    def derivative_batch(self, state, action, dt=None, CT=0):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''       
        Fu = np.array([self.derivative(s, a, dt=dt, CT=CT) for s, a in zip(state, action)])

        return tf.convert_to_tensor(Fu, dtype=tf.float32)
    
    def get_end_effector_position(self, state, end_effector_frame_id=None, recompute=True):
        ''' Compute end-effector position '''
        if end_effector_frame_id is None:
            end_effector_frame_id = self.conf.end_effector_frame_id

        q = state[:self.nq] 

        RF = self.conf.robot.model.getFrameId(end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p
    
    def bound_control_cost(self, action):
        u_cost = sum(action*action + self.conf.w_b*(action/self.conf.u_max)**8) #np.log(1/(action-self.conf.u_min*10)) + np.log(1/(-action+self.conf.u_max*10))) #+ self.conf.w_b*(action[i]/self.conf.u_max[i])**8

        return u_cost

class SingleIntegrator(Env):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):
    
        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE

        self.nx = conf.nx
        self.nu = conf.na
    
    def derivative(self, state, action, dt=None, CT=0):
        ''' Compute the derivative '''
        if dt is None:
            dt = self.conf.dt
        # Dynamics gradient w.r.t control (1st order euler) 
        Fu = np.zeros((self.nx+1, self.nu))
        Fu[0,0] = 1
        Fu[1,1] = 1

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  

        if CT:
            return Fu
        
        Fu[:self.nx, :] *= self.conf.dt

        return Fu
    
    def augmented_derivative(self, state, action):
        ''' Partial derivatives of system dynamics w.r.t. x '''       
        # Compute Jacobians for discrete time dynamics
        Fx = np.zeros((self.conf.nb_state-1,self.conf.nb_state-1))
        Fu = np.zeros((self.conf.nb_state-1,self.conf.nb_action))

        Fx[:self.conf.nb_state-1,:self.conf.nb_state-1] = np.array([[1, 0],
                                                                    [0, 1]])
        
        Fu[0,0] = self.conf.dt
        Fu[1,1] = self.conf.dt
        
        return Fx, Fu
    
    def simulate(self, state, action, dt=None):
        ''' Simulate dynamics '''
        if dt is None:
            dt = self.conf.dt
            
        state_next = np.zeros(self.nx+1)

        state_next[0] = state[0] + self.conf.dt*action[0]
        state_next[1] = state[1] + self.conf.dt*action[1]
        state_next[2] = state[2] + self.conf.dt

        return state_next

    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        p = np.zeros(3)
        p[:2] = state[:2]
        
        return p
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = self.get_end_effector_position(state)[:2]

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        # Term pensalizing the control effort
        if action is not None:
            u_cost = self.bound_control_cost(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset) 

        return r
    
    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.array([self.reward(w, s) for w, s in zip(weights, state)])

        # Redefine action-related cost in tensorflow version
        u_cost = tf.reduce_sum((action**2 + self.conf.w_b*(action/self.conf.u_max)**8),axis=1) 

        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])

class DoubleIntegrator(Env):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):

        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = [self.get_end_effector_position(state)[i] for i in range(2)]

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = np.math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        if action is not None:
            u_cost = self.bound_control_cost(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset)
        
        return r
    
    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.array([self.reward(w, s) for w, s in zip(weights, state)])

        # Redefine action-related cost in tensorflow version
        u_cost = tf.reduce_sum((action**2 + self.conf.w_b*(action/self.conf.u_max)**8),axis=1) 

        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])

class Car(Env):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):
    
        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE

        self.nx = conf.nx
        self.nu = conf.na
    
    def derivative(self, state, action, dt=None, CT=0):
        ''' Compute the derivative '''
        if dt is None:
            dt = self.conf.dt
        # Dynamics gradient w.r.t control (1st order euler) 
        Fu = np.zeros((self.nx+1, self.nu))
        Fu[2,0] = 1
        Fu[4,1] = 1

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  

        if CT:
            return Fu
        
        Fu[:self.nx, :] *= self.conf.dt

        return Fu
    
    def augmented_derivative(self, state, action):
        ''' Partial derivatives of system dynamics w.r.t. x '''       
        # Compute Jacobians for discrete time dynamics
        Fx = np.zeros((self.conf.nb_state-1,self.conf.nb_state-1))
        Fu = np.zeros((self.conf.nb_state-1,self.conf.nb_action))

        Fx[:self.conf.nb_state-1,:self.conf.nb_state-1] = np.array([[1, 0, -self.conf.dt*state[3]*math.sin(state[2]) -self.conf.dt**2*state[4]*math.sin(state[2])/2, self.conf.dt*math.cos(state[2]), self.conf.dt**2*math.cos(state[2])/2], 
                                                                    [0, 1,  self.conf.dt*state[3]*math.cos(state[2]) +self.conf.dt**2*state[4]*math.cos(state[2])/2, self.conf.dt*math.sin(state[2]), self.conf.dt**2*math.sin(state[2])/2], 
                                                                    [0, 0, 1,                                                                                                                      0,                               0], 
                                                                    [0, 0, 0,                                                                                                                      1,                               self.conf.dt], 
                                                                    [0, 0, 0,                                                                                                                      0,                               1]])
        
        Fu[2,0] = self.conf.dt
        Fu[4,1] = self.conf.dt

        return Fx, Fu
    
    def simulate(self, state, action, dt=None):
        ''' Simulate dynamics '''
        if dt is None:
            dt = self.conf.dt

        state_next = np.zeros(self.nx+1)

        state_next[0] = state[0] + self.conf.dt*state[3]*tf.cos(state[2]) + self.conf.dt**2*state[4]*tf.cos(state[2])/2
        state_next[1] = state[1] + self.conf.dt*state[3]*tf.sin(state[2]) + self.conf.dt**2*state[4]*tf.sin(state[2])/2
        state_next[2] = state[2] + self.conf.dt*action[0]
        state_next[3] = state[3] + self.conf.dt*state[4]
        state_next[4] = state[4] + self.conf.dt*action[1]
        state_next[5] = state[5] + self.conf.dt

        return state_next

    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        p = np.zeros(3)
        p[:2] = state[:2]
        
        return p
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = self.get_end_effector_position(state)[:2]

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        # Term pensalizing the control effort
        if action is not None:
            u_cost = self.bound_control_cost(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset) 

        return r
    
    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.array([self.reward(w, s) for w, s in zip(weights, state)])

        # Redefine action-related cost in tensorflow version
        u_cost = tf.reduce_sum((action**2 + self.conf.w_b*(action/self.conf.u_max)**8),axis=1) 
        
        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])

class Manipulator(Env):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):
    
        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE

    def bound_control_cost(self, action):
        u_cost = sum(action*action + self.conf.w_b*(action/self.conf.u_max)**6)
        #u_cost += sum(action*action + self.conf.w_b*(np.exp(-(action-self.conf.u_min)) + np.exp(-(self.conf.u_max-action)) - 2*np.exp(-(self.conf.u_max))))
        
        return u_cost
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = [self.get_end_effector_position(state)[i] for i in range(2)]

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        # Term penalizing the FINAL joint velocity
        if weights[2] != 0:
            vel_cost = state[self.nq:self.nx].dot(state[self.nq:self.nx])
        else:
            vel_cost = 0

        if action is not None:
            u_cost = self.bound_control_cost(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[2]*vel_cost - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset) #- weights[2]*vel_cost 

        return r

    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.array([self.reward(w, s) for w, s in zip(weights, state)])

        # Redefine action-related cost in tensorflow version
        u_cost = tf.reduce_sum((action**2 + self.conf.w_b*(action/self.conf.u_max)**6),axis=1) 
        #u_cost = tf.reduce_sum(action**2 + self.conf.w_b*(tf.math.exp(-(action-self.conf.u_min)) + tf.math.exp(-(self.conf.u_max-action)) - 2*tf.math.exp(-(self.conf.u_max-0*action))),axis=1) 
    
        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])
