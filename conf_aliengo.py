import numpy as np
import jax.numpy as jnp

system_id='aliengo'

''' CACTO parameters '''
NUPDATES = 750001 - 1                                              # Max NNs updates
UPDATE_LOOPS = np.clip(np.arange(1000, 5000000, 3000), 0, 1.5e4)   # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
EP_UPDATE = 10000                                                  # Number of episodes before updating critic and actor
NLOOPS = len(UPDATE_LOOPS)                                         # Number of algorithm loops
NSTEPS = 10                                                        # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                        # Learning rate for the critic network
STD_CRITIC_LEARNING_RATE = 2*CRITIC_LEARNING_RATE                  # Learning rate for the critic network (std)
ACTOR_LEARNING_RATE = 1e-3                                         # Learning rate for the policy network
REPLAY_SIZE = 2**16                                                # Size of the replay buffer
BATCH_SIZE = 512                                                   # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                             # Flag to use MC or TD(n)                                                                                                                                     # Flag to use MC or TD(n)
nsteps_TD_N = NSTEPS                                               # Flag to use MC or TD(n)
UPDATE_RATE = 0
if not MC: 
    UPDATE_RATE = 0.001                                            # Rate to update the target critic network if TD(n) is used
    nsteps_TD_N = int(NSTEPS)                                      # n for the TD(n) update



### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  5000                                          # Save NNs interval
else:
    save_interval = np.inf                                         # Save NNs interval

plot_flag = 1
if plot_flag:
    plot_rollout_interval_diff_loc = 10000                         # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval_diff_loc = np.inf                        # plot.rollout() interval - diff_loc (# update)



''' Cost function parameters '''
### Obstacles parameters
# Default Walls
YC2 = 2
XC2 = -2
B2 = 0.5
A2  = 6

YC1 = 0
XC1 = 1
B1  = 4
A1  = 0.5

YC4 = -2
XC4 = -2
B4  = 0.5
A4  = 6

YC3 = 0
XC3 = -5
B3  = 4
A3  = 0.5

# Default obstacle
YC5 = 0
XC5 = -1
B5  = 1
A5  = 1

### SoftMax parameters 
alpha_obs = 50                       # Soft abs coefficient (obstacle) 
alpha_peak = 5                       # Soft abs coefficient (peak)

### Weights
wc = 2e2                             # CoM position error squared cost weight
wdc = 1e1                            # CoM velocity error squared cost weight
wp = 5e2                             # footstep distance to hip cost weight
wa = 1e2                             # CoP error squared cost weight
wg = 1e2
wb = 1e2                             # Control bound weight
wdt = 1e2                            # Contact phase duration squared cost weight   
w_obs = 5e3                          # Obstacle cost weight         
w_wall = 1e3                         # Wall cost weight       
w_b = 1e2                            # Bound cost weight  
w_peak = 5e3                         # Peak cost weight
w_omega = 1e2                        # Angular velocity squared cost weight

### Gait parameters
foot_step_0 = ["FL", "RR"]           # initial foot steps on the ground

### Cost function parameters
scale_cost_fun = 1e-2

room_size = 10                       # max size of the room
mu = 0.5                             # floor friction coefficient

### Default target parameters
x_des = 0.0                                                                                                # Target x position
y_des = 0.0                                                                                                # Target y position
TARGET_STATE = np.array([x_des, y_des])                                                                    # Target position

maxiter = 1000                                                                                             # Max TO iterations
maxiter_ws = 1000
quant_maxiter = 1
max_cost = 1e5                                                                                             # Max TO cost 



''' Path parameters '''
test_set = 'set test - 1'                                                                    # Test id  
Config_path = './Results Aliengo/Results {}/Configs/'.format(test_set)                            # Configuration path
Fig_path = './Results Aliengo/Results {}/Figures'.format(test_set)                                # Figure path
NNs_path = './Results Aliengo/Results {}/NNs'.format(test_set)                                    # NNs path
Log_path = './Results Aliengo/Results {}/Log/'.format(test_set)                                   # Log path
Code_path = './Results Aliengo/Results {}/Code/'.format(test_set)                                 # Code path
path_list = [Fig_path, NNs_path, Log_path, Code_path]                                             # Path list



''' System parameters '''
nq = None
nv = None
nx = 4 + 2 + 2 + 2 + 4 + 1 + 1
na = 4 + 1 + 1

m = 20                                      # Total mass ( pin.computeTotalMass(model,data) )
Iz = 0.270                                  # Base inertia z ( model.inertias[0].inertia[-1,-1] )
g = 9.81                                    # Gravity vector norm
h = 0.5                                     # Fixed CoM height
w = np.sqrt(g/h)                            # Natural frequency
lx_hip = 0.2407                             # Distance from CoM to hip joint in x direction
ly_hip = 0.051                              # Distance from CoM to hip joint in y direction
lx_tot = 0.31                               # Distance from CoM to the farthest point in x direction
ly_tot = 0.15                               # Distance from CoM to the farthest point in y direction
hip_0 = np.array([ lx_tot, -(ly_tot)])      # hip position in the front left
hip_1 = np.array([-lx_tot,   ly_tot])       # hip position in the front right
hip_2 = np.array([ lx_tot,   ly_tot])       # hip position in the rear left
hip_3 = np.array([-lx_tot, -(ly_tot)])      # hip position in the rear right
hip_mid_0 = np.array([0, -(ly_tot)])        # mid hip position in the front
hip_mid_1 = np.array([0,   (ly_tot)])       # mid hip position in the rear
hip_mid_2 = np.array([ lx_tot, 0])          # mid hip position in the left
hip_mid_3 = np.array([-lx_tot, 0])          # mid hip position in the right
hip_mid = np.array([hip_mid_0, hip_mid_1, hip_mid_2, hip_mid_3])
vx_max = 0.7
vy_max = 0.3

### State parameters 
# state: [feet offset w.r.t. the hip in the current contact phase, CoM position, CoM velocity, obstacle position, offset walls w.r.t. origin, phase]
nb_state = nx + 1
x_init_min = np.array([-0.1, -0.05, -0.1, -0.05, -9.0, -9, -0.7, -0.3, -9, -9, 1, 1, -9, -9, 0.0, 0.0, 0.0])  # State lower bound initial configuration array
x_init_max = np.array([ 0.1,  0.05,  0.1,  0.05,  9.0,  9,  0.7,  0.3,  9,  9, 9, 9, 10, 10, 0.0, 0.0, 0.0])  # State upper bound initial configuration array

# Normalizzation parameters
shift = jnp.zeros(nb_state)
state_norm_arr = jnp.ones(nb_state)

### Action parameters
# control: [feet offset w.r.t. the hip in the next contact phase, CoP coefficient, contact phase duration]
nb_action = na                                                                                           # Action size
dt_min = 0.25                                                                                            # Minimum contact phase duration
dt_max = 0.5                                                                                             # Maximum contact phase duration    
dt_delta = (dt_min+dt_max)/2                                                                             # Delta time for the contact phase duration
u_min = np.array([-0.1, -0.05, -0.1, -0.05, (0.4-0.5), (dt_min - dt_delta)])           # Action lower bound vector
u_max = np.array([ 0.1,  0.05,  0.1,  0.05, (0.6-0.5), (dt_max - dt_delta)])           # Action upper bound vector

stabilizing_controller = np.array([0, 0, 0, 0, 0, -dt_delta])

# initial configurations for plot.rollout()
init_states_sim = []
y_i = np.array([ -1, -0.5, 0.0, 0.5, 1.0])
x_i = np.array([ -4, -3, -2, -1, 0.0])
for i in x_i:
    for j in y_i:
        init_states_sim.append(np.array([0.0, 0.0, 0.0, 0.0, i, j, 0.0, 0.0, XC5, YC5, XC1, YC2, XC3, YC4, ((i-XC5)**2+(j-YC5)**2)**0.5, ((i)**2+(j)**2)**0.5, 0.0]))



### Plot parameters
fig_ax_lim = np.array([[-5, 1],[-2, 2]])                                                                    # Figure axis limit [x_min, x_max, y_min, y_max]
fig_ax_lim2 = np.array([[-9, 9], [-9, 9]])  


profile = 0                                                                                                 # Profile flag


### NNs parameters
critic_type = 'sine'                                                                                        # Activation function - critic (either relu, elu, sine, sine-elu)

features_critic = [64, 64, 128, 128, 1]                                                                     # Features critic and std-critic networks
features_actor = [256, 256, nb_action]                                                                      # Features actor network

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

areg = 1e-3
creg = 1e-2




remapped_indices = None 




tau2u = 0
nb_MPC_step = 2
dt = 1