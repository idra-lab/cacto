import os
import numpy as np
import jax.numpy as jnp
import pinocchio.casadi as cpin
from robot_utils import RobotWrapper, RobotSimulator

system_id = 'manipulator'

''' CACTO parameters '''
NUPDATES = 3000001                                                                                         # Max NNs updates
UPDATE_LOOPS = np.clip(np.arange(1000, 2000000, 3000), 0, 1.5e4)                                           # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
EP_UPDATE = 550                                                                                            # Number of episodes before updating critic and actor
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS))                                                               # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                                 # Number of algorithm loops
NSTEPS = 100                                                                                               # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                                                # Learning rate for the critic network
STD_CRITIC_LEARNING_RATE = 2*CRITIC_LEARNING_RATE                                                          # Learning rate for the std-critic network
ACTOR_LEARNING_RATE = 1e-3                                                                                 # Learning rate for the policy network
REPLAY_SIZE = 2**17                                                                                        # Size of the replay buffer
BATCH_SIZE = 1024#2048                                                                                          # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                                                                     # Flag to use MC or TD(n)
nsteps_TD_N = NSTEPS                                                                                       # Flag to use MC or TD(n)
UPDATE_RATE = 0
if not MC:
    UPDATE_RATE = 0.001                                                                                    # Homotopy rate to update the target critic network
    nsteps_TD_N = int(NSTEPS/2)                                                                            # Number of lookahed steps if TD(n) is used


### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  15000                                                                                  # Save NNs interval
else:
    save_interval = np.inf                                                                                  # Save NNs interval

plot_flag = 1
if plot_flag:
    plot_rollout_interval_diff_loc = 24000                                                                  # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval_diff_loc = np.inf                                                                 # plot.rollout() interval - diff_loc (# update)



''' Cost function parameters '''
### Obstacles parameters
XC1 = -2.0                                                                                                  # X coord center ellipse 1
YC1 = 0.0                                                                                                   # Y coord center ellipse 1
A1  = 6                                                                                                     # Width ellipse 1 
B1  = 10                                                                                                    # Height ellipse 1 
XC2 = 3.0                                                                                                   # X coord center ellipse 2 
YC2 = 4.0                                                                                                   # Y coord center ellipse 2
A2  = 12                                                                                                    # Width ellipse 2 
B2  = 4                                                                                                     # Height ellipse 2 
XC3 = 3.0                                                                                                   # X coord center ellipse 2 
YC3 = -4.0                                                                                                  # Y coord center ellipse 2
A3  = 12                                                                                                    # Width ellipse 2 
B3  = 4                                                                                                     # Height ellipse 2 
obs_param = np.array([XC1, YC1, XC2, YC2, XC3, YC3, A1, B1, A2, B2, A3, B3])                                # Obstacle parameters vector

### Weigths
w_d = 100                                                                                                   # Distance from target weight
w_u = 1e1                                                                                                   # Control effort weight
w_peak = 5e5                                                                                                # Target threshold weight
w_ob = 1e6                                                                                                  # Obstacle weight
w_v = 0                                                                                                     # Velocity weight
w_b = 1e2                                                                                                   # Control bound weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])                                                            # Weights vector (tmp)
cost_weights_running  = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, w_u])                                  # Running cost weights vector
cost_weights_terminal = np.array([w_d, w_peak, w_v, w_ob, w_ob, w_ob, 0])                                   # Terminal cost weights vector 

### SoftMax parameters 
alpha = 50                                                                                                  # Soft abs coefficient (obstacle) 
alpha2 = 5                                                                                                  # Soft abs coefficient (peak)
alpha2_0 = 30
alpha2_1 = 5
soft_max_param = np.array([alpha, alpha2])                                                                  # Soft parameters vector

### Cost function parameters
offset_cost_fun = 0                                                                                         # Reward/cost offset factor
scale_cost_fun = 1e-5                                                                                       # Reward/cost scale factor (1e-5)                                                                       
cost_funct_param = np.array([offset_cost_fun, scale_cost_fun])

### Target parameters
x_des = -20.0                                                                                               # Target x position
y_des = 0.0                                                                                                 # Target y position
TARGET_STATE = np.array([x_des,y_des])                                                                      # Target position

maxiter = 125                                                                                               # Max TO iterations - naive warm-start
maxiter_ws = 50                                                                                             # Max TO iterations - CACTO warm-start
quant_maxiter = 1
max_cost = 1e3 


''' Path parameters '''
test_set = 'set test - 1'                                                                                    # Test id  
Config_path = './Results Manipulator/Results {}/Configs/'.format(test_set)                                   # Configuration path
Fig_path = './Results Manipulator/Results {}/Figures'.format(test_set)                                       # Figure path
NNs_path = './Results Manipulator/Results {}/NNs'.format(test_set)                                           # NNs path
Log_path = './Results Manipulator/Results {}/Log/'.format(test_set)                                          # Log path
Code_path = './Results Manipulator/Results {}/Code/'.format(test_set)                                        # Code path
path_list = [Fig_path, NNs_path, Log_path, Code_path]                                                        # Path list



''' System parameters ''' 
### Robot upload data
URDF_FILENAME = "planar_manipulator_3dof.urdf" 
modelPath = os.getcwd()+"/urdf/" + URDF_FILENAME  
robot = RobotWrapper.BuildFromURDF(modelPath, [modelPath])
nq = robot.nq
nv = robot.nv
nx = nq + nv
na = robot.na
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()
end_effector_frame_id = 'EE'

### Dynamics parameters
dt = 0.05                                                                                                   # Timestep   

simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(robot.na)                                                                       # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'

q_init, v_init = np.array([0, 0, np.pi]), np.zeros(robot.nv)
simu = RobotSimulator(robot, q_init, v_init, simulation_type, tau_coulomb_max)

### System configuration parameters
x_base = -7.0                                                                                               # x coord base
y_base = 0.0 
l = 10                                                                                                      # y coord base

### State parameters 
# state: theta_1, theta_2, theta_3, omega_1, omega_2, omega_3, t
nb_state = robot.nq + robot.nv + 1                                                                          # State size (robot state size +1)
x_min = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -np.inf, 0])                           # State lower bound vector
x_init_min = np.array([-jnp.pi, -jnp.pi, -jnp.pi, -jnp.pi/4, -jnp.pi/4, -jnp.pi/4, 0])                      # State lower bound initial configuration vector
x_max = jnp.array([ jnp.inf,  jnp.inf,  jnp.inf,  jnp.inf,  jnp.inf,  jnp.inf, jnp.inf])                    # State upper bound vector
x_init_max = jnp.array([ jnp.pi,  jnp.pi,  jnp.pi,  jnp.pi/4,  jnp.pi/4,  jnp.pi/4, 0*(NSTEPS)*dt])         # State upper bound initial configuration vector
shift = jnp.array([0, 0, 0, 0, 0, 0, -int(NSTEPS*dt)/2])             
state_norm_arr = jnp.array([1,1,1,jnp.pi/4,jnp.pi/4,jnp.pi/4,int(NSTEPS*dt)])                               # Array used to normalize states

### Action parameters
# control: tau_1, tau_2, tau_3
nb_action = robot.na                                                                                        # Action size
tau_lower_bound = -200                                                                                      # Action lower bound
tau_upper_bound = 200                                                                                       # Action upper bound
u_min = tau_lower_bound*np.ones(nb_action)                                                                  # Action lower bound vector
u_max = tau_upper_bound*np.ones(nb_action)                                                                  # Action upper bound vector
stabilizing_controller = np.zeros(nb_action)                                                                # Naive initial guess

# initial configurations for plot.rollout()
init_states_sim = [np.array([jnp.pi/4,    -jnp.pi/8, -jnp.pi/8, 0.0, 0.0, 0.0, 0.0]),                             
                   np.array([-jnp.pi/4,   jnp.pi/8,  jnp.pi/8,  0.0, 0.0, 0.0, 0.0]),
                   np.array([jnp.pi/2,    0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-jnp.pi/2,   0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([3*jnp.pi/4,  0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-3*jnp.pi/4, 0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([jnp.pi/4,    0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-jnp.pi/4,   0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([jnp.pi,      0.0,        0.0,        0.0, 0.0, 0.0, 0.0]),
                   np.array([-1.55135003,  2.93707696, -1.3025857 , 0., 0., 0., 0. ]),
                   np.array([ 1.55135003, -2.93707696,  1.3025857 , 0., 0., 0., 0. ]),
                   np.array([-1.31811607,  2.63623214, -1.31811607, 0., 0., 0., 0. ]),
                   np.array([-0.98843209,  1.97686418, -0.98843209, 0., 0., 0., 0. ])]


### Plot parameters
fig_ax_lim = np.array([[-41, 31], [-35, 35]])                                                               # Figure axis limit [x_min, x_max, y_min, y_max]



profile = 0                                                                                                 # Profile flag
supervise_learn_flag = 0                                                                                    # Supervised flag



### NNs parameters
features_critic = [64, 64, 128, 128, 1]                                                                     # Features critic and std-critic networks
features_actor = [256, 256, nb_action]                                                                      # Features actor network

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

areg = 1e-3
creg = 1e-1

remap_indices = jnp.array([0,1,2])
base = jnp.arange(nb_state)
counts = jnp.where(jnp.isin(base, remap_indices), 2, 1)
remapped_indices = jnp.repeat(base, counts)
n_before = jnp.arange(len(remap_indices))
cos_element = remap_indices + n_before
sin_element = remap_indices + n_before + 1


tau2u = 1
nb_MPC_step = 1