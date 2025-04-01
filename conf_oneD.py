import math
import numpy as np

system_id = 'oneD'

''' CACTO parameters '''
NUPDATES = 20000                                                                                           # Max NNs updates
UPDATE_LOOPS = np.clip(np.arange(1000, 100000, 3000), 0, 1.5e4)                                                                 # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                           
EP_UPDATE = 50 #*np.ones(len(UPDATE_LOOPS))                                                                                             # Number of episodes before updating critic and actor
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS)) #int(sum(EP_UPDATE)                                                                # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                                  # Number of algorithm loops
NSTEPS = 50                                                                                                # Max episode length
CRITIC_LEARNING_RATE = 1e-4                                                                                 # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                                  # Learning rate for the policy network
REPLAY_SIZE = 2**15                                                                                         # Size of the replay buffer
BATCH_SIZE = 64                                                                                             # Size of the mini-batch 

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                                                                      # Flag to use MC or TD(n)
if not MC:
    UPDATE_RATE = 0.001                                                                                     # Homotopy rate to update the target critic network if TD(n) is used
    nsteps_TD_N = int(NSTEPS/4)                                                                             # Number of lookahed steps if TD(n) is used


### Savings parameters
save_flag = 1
if save_flag:
    save_interval =  10000                                                                                  # Save NNs interval
else:
    save_interval = np.inf                                                                                  # Save NNs interval

plot_flag = 1
if plot_flag:
    plot_rollout_interval = 400                                                                             # plot.rollout() interval (# update)
    plot_rollout_interval_diff_loc = 6000                                                                   # plot.rollout() interval - diff_loc (# update)
else:
    plot_rollout_interval = np.inf                                                                          # plot.rollout() interval (# update)
    plot_rollout_interval_diff_loc = np.inf                                                                 # plot.rollout() interval - diff_loc (# update)



### NNs parameters
critic_type = 'sine'                                                                                        # Activation function - critic (either relu, elu, sine, sine-elu)

NH1 = 256                                                                                                   # 1st hidden layer size - actor
NH2 = 256                                                                                                   # 2nd hidden layer size - actor

LR_SCHEDULE = 0                                                                                             # Flag to use a scheduler for the learning rates
boundaries_schedule_LR_C = [200*REPLAY_SIZE/BATCH_SIZE, 
                            300*REPLAY_SIZE/BATCH_SIZE,
                            400*REPLAY_SIZE/BATCH_SIZE,
                            500*REPLAY_SIZE/BATCH_SIZE]     
# Values of critic LR                            
values_schedule_LR_C = [CRITIC_LEARNING_RATE,
                        CRITIC_LEARNING_RATE/2,
                        CRITIC_LEARNING_RATE/4,
                        CRITIC_LEARNING_RATE/8,
                        CRITIC_LEARNING_RATE/16]  
# Numbers of critic updates after which the actor LR is changed (based on values_schedule_LR_A)
boundaries_schedule_LR_A = [200*REPLAY_SIZE/BATCH_SIZE,
                            300*REPLAY_SIZE/BATCH_SIZE,
                            400*REPLAY_SIZE/BATCH_SIZE,
                            500*REPLAY_SIZE/BATCH_SIZE]   
# Values of actor LR                            
values_schedule_LR_A = [ACTOR_LEARNING_RATE,
                        ACTOR_LEARNING_RATE/2,
                        ACTOR_LEARNING_RATE/4,
                        ACTOR_LEARNING_RATE/8,
                        ACTOR_LEARNING_RATE/16]  

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

kreg_l1_A = 1e-2                                                                                            # Weight of L1 regularization in actor's network - kernel
kreg_l2_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - kernel
breg_l1_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - bias
breg_l2_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - bias
kreg_l1_C = 1e-2                                                                                            # Weight of L1 regularization in critic's network - kernel
kreg_l2_C = 1e-2                                                                                            # Weight of L2 regularization in critic's network - kernel
breg_l1_C = 1e-2                                                                                            # Weight of L1 regularization in critic's network - bias
breg_l2_C = 1e-2                                                                                            # Weight of L2 regularization in critic's network - bias

### Buffer parameters
prioritized_replay_alpha = 0                                                                                # α determines how much prioritization is used, set to 0 to use a normal buffer. Used to define the probability of sampling transition i --> P(i) = p_i**α / sum(p_k**α) where p_i is the priority of transition i 
prioritized_replay_beta = 0.6          
prioritized_replay_beta_iters = None                                                                        # Therefore let's exploit the flexibility of annealing the amount of IS correction over time, by defining a schedule on the exponent β that from its initial value β0 reaches 1 only at the end of learning.
prioritized_replay_eps = 1e-2                                                                               # It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
fresh_factor = 0.95                                                                                         # Refresh factor



''' Cost function parameters '''
### Weigths
w_u = 1e-6                                                                                                    # Control effort weight
cost_weights_running  = np.array([w_u])                                  # Running cost weights vector
cost_weights_terminal = np.array([0])                                    # Terminal cost weights vector 



''' Path parameters '''
test_set = 'set test check 2'                                                                                       # Test id  
Config_path = './Results oneD/Results {}/Configs/'.format(test_set)                                          # Configuration path
Fig_path = './Results oneD/Results {}/Figures'.format(test_set)                                              # Figure path
NNs_path = './Results oneD/Results {}/NNs'.format(test_set)                                                  # NNs path
Log_path = './Results oneD/Results {}/Log/'.format(test_set)                                                 # Log path
Code_path = './Results oneD/Results {}/Code/'.format(test_set)                                               # Code path
DictWS_path = './Results oneD/Results {}/DictWS/'.format(test_set)                                           # DictWS path
path_list = [Fig_path, NNs_path, Log_path, Code_path, DictWS_path]                                          # Path list

# Recover-training parameters
test_set_rec = None
NNs_path_rec = './Results oneD/Results set {}/NNs'.format(test_set_rec)                                      # NNs path recover training
N_try_rec = None
update_step_counter_rec = None

''' System parameters ''' 
env_RL = 0                                                                                                  # Flag RL environment: set True if RL_env and TO_env are different                                                                                 

### Dynamics parameters
dt = 0.05                                                                                                   # Timestep   

simulate_coulomb_friction = 0                                                                               # To simulate friction
simulation_type = 'euler'                                                                                   # Either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(2)                                                                              # Expressed as percentage of torque max
integration_scheme = 'E-Euler'                                                                              # TO integration scheme - Either 'E-Euler' or 'SI-Euler'                                                                             # Link mass

### State parameters 
nb_state = 1 + 1                                                                                            # State size (robot state size +1)
nq = None
nv = None
nx = 1
na = 1
x_min = np.array([-np.inf, 0])                                          # State lower bound vector
x_init_min = np.array([-2, 0])                                                     # State lower bound initial configuration array
x_max = np.array([np.inf, np.inf])                                          # State upper bound vector
x_init_max = np.array([ 2,  (NSTEPS-1)*dt])                                           # State upper bound initial configuration array
state_norm_arr = np.array([2, int(NSTEPS*dt)])                                         # Array used to normalize states
# state: x, y, theta, v, a, t

# initial configurations for plot.rollout()
init_states_sim = [np.array([-2.0,  0.0]),
                   np.array([-1.5, 0.0]),
                   np.array([-1.0, 0.0]),
                   np.array([-0.5, 0.0]),
                   np.array([0,0.0]),
                   np.array([0.5,0.0]),
                   np.array([1.0, 0.0]),
                   np.array([1.5, 0.0]),
                   np.array([2.0, 0.0])]

### Action parameters
nb_action = 1                                                                                               # Action size
bound_actions = 0
u_lower_bound = -1                                                                                      # Action lower bound
u_upper_bound = 1     
u_min = np.array([u_lower_bound])                                                     # Action lower bound vector
u_max = np.array([u_upper_bound])                                                     # Action upper bound vector
w_b = 1/w_u



### Plot parameters
fig_ax_lim = np.array([[-2, 2], [-6, 2]])                                                               # Figure axis limit [x_min, x_max, y_min, y_max]



profile = 0                                                                                                 # Profile flag