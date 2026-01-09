import jax
import functools
import jax.numpy as jnp
from jax import lax
from jaxadi import convert
from trajax import optimizers
jax.config.update("jax_enable_x64", False)

def wrap_convert(f):
    """Convert a CasADi function into a JAX-compatible and JIT-compiled function."""
    f_converted = convert(f)

    @jax.jit
    def wrapped(*args):
        # Squeeze the first element (CasADi returns a tuple)
        return jnp.squeeze(f_converted(*args)[0])

    return wrapped

class TO_JAX:
    def __init__(self, env_class, conf):
        """
        Trajectory Optimization class.
        
        Attributes: 
            env : (environment_TO class instance) Environment class instance. Includes the following methods:
                dynamics_tau_func_wrapped :                                       (callable) Compute next state
                cost_tau_func_wrapped :                                           (callable) Compute cost
                p_ee_jax_wrapped :                                                (callable) Compute end effector position
                get_u_from_tau_func_wrapped (only if URDF-based system is used) : (callable) Compute acceleration given torque 
                dynamics_func_wrapped (only if URDF-based system is used) :       (callable) Compute next state with inverse dynamics
                cost_func_wrapped (only if URDF-based system is used) :           (callable) Compute cost with inverse dynamics                

            conf : (object) Configuration object containing the following attributes:
                stabilizing_controller :                 (float array) Stable warm-start control
                tau2u :                                  (bool) Whether to map torque to acceleration 
                nb_MPC_step (only if MPC mode is used) : (int) Number of Model Predictive Control (MPC) steps.
                NSTEPS :                                 (int) Maximum horizon' lenght
                nb_state :                               (int) Dimension of the system state vector.
                nb_action :                              (int) Dimension of the system control vector.
                max_cost :                               (int) Maximum total cost to be considered a TO problem converged
                MC :                                     (bool) Whether Monte Carlo or Temporal Difference N-step is used in RL phase
                nsteps_TD_N (only if MC==0):             (int) Number of steps used in TD(N) computation.

        """
        self.env = env_class
        self.conf = conf

    @functools.partial(jax.jit, static_argnames=("self", "actor_model", "init"))
    def create_TO_ws(self, ICS, actor_model, actor_params, init=1):    
        """Create initial warm-start trajectories (state and control sequences) for TO."""
        def single_rollout(x0, init):
            """Perform a single trajectory rollout given an initial state."""
            def dynamics_for_scan(x, _):
                # Choose tau based on init flag: 
                #     stabilize controller for the first iteration (init = 0)
                #     actor's output for the following iterations
                tau = lax.cond(
                    init == 0,
                    lambda _: self.conf.stabilizing_controller,
                    lambda _: actor_model(actor_params, x),
                    operand=None,
                )

                # Compute next state and control
                x_next = self.env.dynamics_tau_func_wrapped(x, tau)
                # Convert tau into u if required (inverse dynamics for URDF-based systems in TO)
                if self.conf.tau2u:
                    u = self.env.get_u_from_tau_func_wrapped(x, tau.reshape(-1, 1))
                else:
                    u = tau

                # Reset time-state if MPC is used (init = -1)
                x_next = lax.cond(
                    init == -1,
                    lambda _: jnp.concatenate((x_next[:-1], jnp.array([x_next[-1] % self.conf.nb_MPC_step]))),
                    lambda _: x_next,
                    operand=None,
                )

                return x_next, (x_next, u)

            # Rollout across NSTEPS
            _, (X, U) = lax.scan(dynamics_for_scan, x0, jnp.arange(self.conf.NSTEPS))

            # Include initial state in the trajectory
            X = jnp.vstack((x0, X))

            return X, U

        # Vectorized rollout over all initial conditions
        X_batch, U_batch = jax.vmap(lambda x0: single_rollout(x0, init))(ICS)

        success_init_flag = 1  # Currently initialization always considered successful

        return X_batch, U_batch, success_init_flag
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def cost_func_varT(self, x, u, t, params):
        """Cost function simulating horizon = T (params)."""
        cost_fn = self.env.cost_func_wrapped if self.conf.tau2u else self.env.cost_tau_func_wrapped # Inverse dynamics used for urdf based systems
        return jnp.where(t <= params, 1, 0)*cost_fn(x,u)
    
    @functools.partial(jax.jit, static_argnums=(0,4,5))
    def single_optimization(self, x0, U_init, T, maxiter, psd_delta):
        """Solve a single trajectory optimization problem using ILQR."""
        
        dyn_fn = self.env.dynamics_func_wrapped if self.conf.tau2u else self.env.dynamics_tau_func_wrapped # Inverse dynamics used for urdf based systems

        # Run ILQR optimization
        X, U, _, _, _, _, iteration, dvdx, cost_arr = optimizers.ilqr(
            functools.partial(self.cost_func_varT, params=T),
            dyn_fn,
            x0,
            U_init,
            maxiter=maxiter,
            psd_delta=psd_delta,
        )
        
        # Compute end-effector position trajectory
        p_ee = jax.vmap(self.env.p_ee_jax_wrapped)(X.reshape(-1, self.conf.nb_state, 1))
        
        # Define success based on cost threshold and iteration limit
        if self.conf.system_id == 'aliengo':
            success_flag = (jnp.sum(cost_arr[:-1]) < self.conf.max_cost) & (iteration < maxiter)# & (1 - jnp.any(jax.vmap(inequality_constraint)(X[:-1], U, X[:-1])>1e-3))
        else:
            success_flag = (jnp.sum(cost_arr) < self.conf.max_cost) & (iteration < maxiter)

        return (
            X,
            U,
            p_ee,
            cost_arr,
            success_flag,
            dvdx,
        )
    
    @functools.partial(jax.jit, static_argnums=(0,4,5))
    def optimize_batch(self, x0_batch, U_batch, T_batch, maxiter, psd_delta):
        return jax.vmap(lambda x0, U, T: self.single_optimization(x0, U, T, maxiter, psd_delta))(
            x0_batch, U_batch, T_batch
        )
    
    def TO_System_Solve(self, x0_batch, U_batch, T_batch, maxiter=1000, psd_delta=1e-3):
        return self.optimize_batch(x0_batch, U_batch, T_batch, maxiter, psd_delta)
    
    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2, 3, 4))
    def postprocess_TO_data(self, TO_states, TO_step_cost, TO_ee_pos_arr, dVdx):
        """Post-process trajectory optimization data for the reinforcement learning phase."""
        nsteps_sh = TO_states.shape[0]
        state_arr = TO_states

        ee_pos_arr = TO_ee_pos_arr
        rwrd_arr = TO_step_cost
        
        # Determine lookahead step indices and done flags
        steps = jnp.arange(nsteps_sh)
        if self.conf.MC:
            # Monte Carlo mode: episode terminates after full horizon
            final_lookahead_step = jnp.full(self.conf.NSTEPS, nsteps_sh)
            done_arr = jnp.ones(nsteps_sh)
        else:
            # TD mode: horizon-limited lookahead (nsteps_TD_N)
            final_lookahead_step = jnp.minimum(steps + self.conf.nsteps_TD_N, nsteps_sh-1)
            done_arr = (steps + self.conf.nsteps_TD_N) >= (nsteps_sh-1)

        # Compute next states for each lookahead
        state_next_rollout_arr = state_arr[(final_lookahead_step + 1).astype(jnp.int32)]

        def compute_cost_to_go(costs: jnp.ndarray):
            """Compute the partial cost-to-go over a fixed horizon."""
            cumsum_costs = jnp.cumsum(costs)
        
            cost_to_go = cumsum_costs[self.conf.nsteps_TD_N:] - jnp.concatenate([jnp.zeros(1), cumsum_costs[:-self.conf.nsteps_TD_N-1]])
            last_sums = jnp.flip(jnp.cumsum(jnp.flip(costs[-self.conf.nsteps_TD_N:]))) 

            return jnp.concatenate([cost_to_go, last_sums])

        # Compute partial cost-to-go
        partial_reward_to_go_arr = compute_cost_to_go(rwrd_arr)

        return state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ee_pos_arr, dVdx