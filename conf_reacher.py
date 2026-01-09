import os
import math
import numpy as np
import jax.numpy as jnp
import pinocchio.casadi as cpin
from robot_utils import RobotWrapper, RobotSimulator

system_id = 'reacher'

''' CACTO parameters '''
NUPDATES = 150001                                                              # Max NNs updates
UPDATE_LOOPS = np.clip(np.arange(1000, 500000, 3000), 0, 1.5e4)                # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
EP_UPDATE = 250                                                                # Number of episodes before updating critic and actor
NLOOPS = len(UPDATE_LOOPS)                                                     # Number of algorithm loops
NSTEPS = 250                                                                   # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                    # Learning rate for the critic network
STD_CRITIC_LEARNING_RATE = 2*CRITIC_LEARNING_RATE                              # Learning rate for the std-critic network
ACTOR_LEARNING_RATE = 1e-3                                                     # Learning rate for the policy network
REPLAY_SIZE = 2**17                                                            # Size of the replay buffer
BATCH_SIZE = 2048                                                              # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0
nsteps_TD_N = NSTEPS                                                           # Flag to use MC or TD(n)
UPDATE_RATE = 0
if not MC:
    UPDATE_RATE = 0.001                                                        # Homotopy rate to update the target critic network if TD(n) is used
    nsteps_TD_N = int(NSTEPS/4)                                                # Number of lookahed steps if TD(n) is used


### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  10000                                                     # Save NNs interval
else:
    save_interval = np.inf                                                     # Save NNs interval

plot_flag = 1
if plot_flag:
    plot_rollout_interval_diff_loc = 24000                                      # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval_diff_loc = np.inf                                     # plot.rollout() interval - diff_loc (# update)



''' Cost function parameters '''
### Weigths
w_d = 100                                                                          # Distance from target weight
w_u = 1e0                                                                          # Control effort weight
w_peak = 5e5                                                                       # Target threshold weight
w_ob = 1e6                                                                         # Obstacle weight
w_v = 0                                                                            # Velocity weight
w_b = 1e2                                                                          # Control bound weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])                                   # Weights vector (tmp)
cost_weights_running  = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, w_u])         # Running cost weights vector
cost_weights_terminal = np.array([w_d, w_peak, w_v, w_ob, w_ob, w_ob, 0])          # Terminal cost weights vector                                                               # Soft parameters vector

### Cost function parameters
offset_cost_fun = 0                                                                # Reward/cost offset factor
scale_cost_fun = 1e-5                                                              # Reward/cost scale factor (1e-5)                                                                       
cost_funct_param = np.array([offset_cost_fun, scale_cost_fun])

### Default target parameters
x_des = 0.1                                                                        # Target x position
y_des = 0.0                                                                        # Target y position
TARGET_STATE = np.array([x_des,y_des])                                             # Target position

maxiter = 75                                                                       # Max TO iterations - naive warm-start
maxiter_ws = 15                                                                    # Max TO iterations - CACTO warm-start
quant_maxiter = 1
max_cost = 1e3



''' Path parameters '''
test_set = 'set test - 1'                                                          # Test id  
Config_path = './Results Reacher/Results {}/Configs/'.format(test_set)             # Configuration path
Fig_path = './Results Reacher/Results {}/Figures'.format(test_set)                 # Figure path
NNs_path = './Results Reacher/Results {}/NNs'.format(test_set)                     # NNs path
Log_path = './Results Reacher/Results {}/Log/'.format(test_set)                    # Log path
Code_path = './Results Reacher/Results {}/Code/'.format(test_set)                  # Code path
path_list = [Fig_path, NNs_path, Log_path, Code_path]                              # Path list



''' System parameters ''' 
### Robot upload data
URDF_FILENAME = "reacher.urdf" 
modelPath = os.getcwd()+"/urdf/" + URDF_FILENAME  
robot = RobotWrapper.BuildFromURDF(modelPath, [modelPath])
robot.model.armature[:] = [1.0, 1.0]
robot.model.damping[:] = [1.0, 1.0]
nq = robot.nq
nv = robot.nv
nx = nq + nv
na = robot.na
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()
end_effector_frame_id = 'EE'

### Dynamics parameters
n_frames = 2
dt = 0.01 * n_frames                                                                                        # Timestep   

simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(robot.na)                                                                       # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'

q_init, v_init = np.array([0, 0]), np.zeros(robot.nv)
simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max)

### System configuration parameters
x_base = 0.0                                                                                            
y_base = 0.0 
l1 = 0.1 
l2 = 0.11                                                                                           

### State parameters 
# state: theta_1, theta_2, omega_1, omega_2, target_x, target_y, ee_x-target_x, ee_y-target_y, t
nb_state = robot.nq + robot.nv + 2 + 2 + 1                                                                     # State size (robot state size +1)
x_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0])                  # State lower bound vector
x_init_min = np.array([-math.pi, -math.pi, -math.pi/4, -math.pi/4,  0, -np.pi, 0, 0, 0])                       # State lower bound initial configuration vector
x_max = np.array([ np.inf,  np.inf, np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf, np.inf])               # State upper bound vector
x_init_max = np.array([ math.pi,  math.pi,  math.pi/4,  math.pi/4,  0.2,  np.pi, 0, 0, 0*(NSTEPS)*dt])         # State upper bound initial configuration vector
shift = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, -int(NSTEPS*dt)/2])             
state_norm_arr = np.array([1,1,jnp.pi/4,jnp.pi/4, 0.2, 0.2, 1, 1, int(NSTEPS*dt)])                             # Array used to normalize states

### Action parameters
nb_action = robot.na                                                                                           # Action size
tau_lower_bound = -25                                                                                          # Action lower bound
tau_upper_bound = 25                                                                                           # Action upper bound
u_min = tau_lower_bound*np.ones(nb_action)                                                                     # Action lower bound vector
u_max = tau_upper_bound*np.ones(nb_action)                                                                     # Action upper bound vector
stabilizing_controller = np.zeros(nb_action)                                                                   # Naive initial guess

# initial configurations for plot.rollout()
init_states_sim = [jnp.array([-0, 0, 0.0, 0.0, 0.10, 0.10, 0.0, 0.0, 0.0]),
                   jnp.array([-0, 0, 0.0, 0.0, 0.10,-0.05, 0.0, 0.0, 0.0]),                             
                   jnp.array([-0, 0, 0.0, 0.0, 0.05, 0.15, 0.0, 0.0, 0.0]),
                   jnp.array([-0, 0, 0.0, 0.0,-0.05, 0.05, 0.0, 0.0, 0.0]),
                   jnp.array([0,  0, 0.0, 0.0,-0.15,-0.1,  0.0, 0.0, 0.0]),
                   jnp.array([-0, 0, 0.0, 0.0,-0.2,  0.0,  0.0, 0.0, 0.0]),
                   jnp.array([0, -0, 0.0, 0.0, 0.0,  0.2,  0.0, 0.0, 0.0]),
                   jnp.array([-0,-0, 0.0, 0.0,-0.1,  0.05, 0.0, 0.0, 0.0])]





### Plot parameters
fig_ax_lim = np.array([[-0.25, 0.25], [-0.25, 0.25]])                                                             # Figure axis limit [x_min, x_max, y_min, y_max]



profile = 0                                                                                                       # Profile flag
supervise_learn_flag = 0                                                                                          # Supervised flag



### NNs parameters
features_critic = [256, 256, 1]                                                                             # Features critic and std-critic networks
features_actor = [256, 256, nb_action]                                                                      # Features actor network

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

areg = 1e-3
creg = 1e-1

remap_indices = jnp.array([0,1])
base = jnp.arange(nb_state)
counts = jnp.where(jnp.isin(base, remap_indices), 2, 1)
remapped_indices = jnp.repeat(base, counts)
n_before = jnp.arange(len(remap_indices))
cos_element = remap_indices + n_before
sin_element = remap_indices + n_before + 1


tau2u = 1
nb_MPC_step = 1