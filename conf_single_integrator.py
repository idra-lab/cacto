import numpy as np

system_id = 'single_integrator'

''' CACTO parameters '''
NUPDATES = 25001 - 1                                                           # Max NNs updates
UPDATE_LOOPS = np.clip(np.arange(1000, 500000, 3000), 0, 1.5e4)                # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
EP_UPDATE = 200                                                                # Number of episodes before updating critic and actor
NLOOPS = len(UPDATE_LOOPS)                                                     # Number of algorithm loops
NSTEPS = 100                                                                   # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                    # Learning rate for the critic network
STD_CRITIC_LEARNING_RATE = 2*CRITIC_LEARNING_RATE                              # Learning rate for the std-critic network
ACTOR_LEARNING_RATE = 1e-3                                                     # Learning rate for the policy network
REPLAY_SIZE = 2**16                                                            # Size of the replay buffer
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
    save_interval =  5000                                                      # Save NNs interval
else:
    save_interval = np.inf                                                     # Save NNs interval

plot_flag = 1
if plot_flag:
    plot_rollout_interval_diff_loc = 6000                                      # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval_diff_loc = np.inf                                    # plot.rollout() interval - diff_loc (# update)



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
w_u = 10                                                                                                    # Control effort weight
w_peak = 5e5                                                                                                # Target threshold weight
w_ob = 5e6                                                                                                  # Obstacle weight
w_v = 0                                                                                                     # Velocity weight
w_b = 1e2                                                                                                   # Control bound weight
weight = np.array([w_d, w_u, w_peak, w_ob, w_v])                                                            # Weights vector (tmp)
cost_weights_running  = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, w_u])                                  # Running cost weights vector
cost_weights_terminal = np.array([w_d, w_peak, 0., w_ob, w_ob, w_ob, 0])                                    # Terminal cost weights vector 

### SoftMax parameters 
alpha = 50                                                                                                  # Soft abs coefficient (obstacle) 
alpha2 = 5                                                                                                  # Soft abs coefficient (peak)
soft_max_param = np.array([alpha, alpha2])                                                                  # Soft parameters vector

### Cost function parameters
offset_cost_fun = 0                                                                                         # Reward/cost offset factor
scale_cost_fun = 1e-5                                                                                       # Reward/cost scale factor (1e-5)                                                                       
cost_funct_param = np.array([offset_cost_fun, scale_cost_fun])

### Target parameters
x_des = -7.0                                                                                                # Target x position
y_des = 0.0                                                                                                 # Target y position
TARGET_STATE = np.array([x_des,y_des])                                                                      # Target position

maxiter = 200                                                                                               # Max TO iterations
max_cost = 1e3                                                                                              # Max TO cost 



''' Path parameters '''
test_set = 'set test - 1'                                                                                   # Test id  
Config_path = './Results Single Integrator/Results {}/Configs/'.format(test_set)                            # Configuration path
Fig_path = './Results Single Integrator/Results {}/Figures'.format(test_set)                                # Figure path
NNs_path = './Results Single Integrator/Results {}/NNs'.format(test_set)                                    # NNs path
Log_path = './Results Single Integrator/Results {}/Log/'.format(test_set)                                   # Log path
Code_path = './Results Single Integrator/Results {}/Code/'.format(test_set)                                 # Code path
path_list = [Fig_path, NNs_path, Log_path, Code_path]                                                       # Path list



''' System parameters ''' 
nq = None
nv = None
nx = 2
na = 2


### Dynamics parameters
dt = 0.05                                                                                                   # Timestep   

simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(na)                                                                             # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'



### State parameters 
nb_state = nx + 1                                                                                           # State size (robot state size +1)
x_min = np.array([-np.inf, -np.inf, 0])                                                                     # State lower bound vector
x_init_min = np.array([-15, -15, 0])                                                                        # State lower bound initial configuration array
x_max = np.array([np.inf, np.inf, np.inf])                                                                  # State upper bound vector
x_init_max = np.array([ 15,  15, 0*(NSTEPS)*dt])                                                            # State upper bound initial configuration array
shift =np.array([0, 0, int(NSTEPS*dt)/2])
state_norm_arr = np.array([15, 15, int(NSTEPS*dt)])                                                         # Array used to normalize states

### Action parameters
nb_action = na                                                                                              # Action size
tau_lower_bound = -6                                                                                        # Action lower bound
tau_upper_bound = 6                                                                                         # Action upper bound
u_min = tau_lower_bound*np.ones(nb_action)                                                                  # Action lower bound vector
u_max = tau_upper_bound*np.ones(nb_action)                                                                  # Action upper bound vector
stabilizing_controller = np.zeros(nb_action)                                                                # Naive initial guess

# initial configurations for plot.rollout()
init_states_sim = [np.array([2.0,   0.0,   0.0]),
                   np.array([10.0,  0.0,   0.0]),
                   np.array([10.0,  -10.0, 0.0]),
                   np.array([10.0,  10.0,  0.0]),
                   np.array([-10.0, 10.0,  0.0]),
                   np.array([-10.0, -10.0, 0.0]),
                   np.array([12.0,  2.0,   0.0]),
                   np.array([12.0,  -2.0,  0.0]),
                   np.array([15.0,  0.0,   0.0])]



### Plot parameters
fig_ax_lim = np.array([[-16, 16], [-16, 16]])                                                               # Figure axis limit [x_min, x_max, y_min, y_max]



profile = 0                                                                                                 # Profile flag
supervise_learn_flag = 0                                                                                    # Supervised flag



### NNs parameters
features_critic = [64, 64, 128, 128, 1]                                                                     # Features critic and std-critic networks
features_actor = [256, 256, nb_action]                                                                      # Features actor network

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

areg = 1e-3
creg = 1e-1

remapped_indices = None

tau2u = 0
nb_MPC_step = 1