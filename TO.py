import casadi
import numpy as np

class TO_Casadi:
    
    def __init__(self, conf, env_TO, w_S=0):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param u_min :                      (float array) Action lower bound array
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param dt :                         (float) Timestep

        :input system_id :                      (str) Id system
        
        :input w_S :                            (float) Sobolev-training weight
        '''
        
        self.conf = conf

        self.nx = conf.nb_state-1
        self.nu = conf.nb_action

        self.w_S = w_S

        self.CAMS = env_TO
    
    def TO_System_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Create and solbe TO casadi problem '''
        ### PROBLEM
        opti = casadi.Opti()

        # The control models are stored as a collection of shooting nodes called running models, with an additional terminal model.
        self.runningSingleModel = self.CAMS('running_model', self.conf)
        runningModels = [ self.runningSingleModel for _ in range(T) ]
        self.terminalModel = self.CAMS('terminal_model', self.conf)
        
        # Decision variables
        xs = [ opti.variable(model.nx) for model in runningModels+[self.terminalModel] ]     # state variable
        us = [ opti.variable(model.nu) for model in runningModels ]                          # control variable
        
        # Roll out loop, summing the integral cost and defining the shooting constraints.
        total_cost = 0
        
        x0 = opti.parameter(len(ICS_state) - 1)
        opti.subject_to(xs[0] == x0)

        for t in range(T):
            x_next, r_cost = runningModels[t].step_fun(xs[t], us[t])
            opti.subject_to(xs[t + 1] == x_next )
            total_cost += r_cost 
        r_cost_final = self.terminalModel.cost(xs[-1], us[-1])
        total_cost += r_cost_final
        
        ### SOLVE
        opti.minimize(total_cost)  
        
        # Create warmstart
        init_x_TO = [np.array(init_TO_states[i,:-1]) for i in range(T+1)]
        init_u_TO = [np.array(init_TO_controls[i,:]) for i in range(T)]

        for x,xg in zip(xs,init_x_TO): opti.set_initial(x,xg)
        for u,ug in zip(us,init_u_TO): opti.set_initial(u,ug)

        # Set solver options
        opts = {'ipopt.sb': 'yes','ipopt.print_level': 0, 'print_time': 0} #,'ipopt.linear_solver':'ma57', 
        opti.solver("ipopt", opts) 
        
        try:
            opti.set_value(x0, ICS_state[:-1])
            opti.solve()
            #TO_states = np.array([[ opti.value(x) for x in xs ]]).T ### DEBUG
            #TO_controls = np.array([[ opti.value(u) for u in us ]]).T            
            TO_states = np.array([ opti.value(x) for x in xs ])
            TO_controls = np.array([ opti.value(u) for u in us ])
            TO_total_cost = opti.value(total_cost)
            TO_ee_pos_arr = np.empty((T+1,3))
            TO_step_cost = np.empty(T+1)
            for n in range(T):
                TO_ee_pos_arr[n,:] = np.reshape(runningModels[n].p_ee(TO_states[n,:]),-1)
                TO_step_cost[n] = runningModels[n].cost(TO_states[n,:], TO_controls[n,:])
            TO_ee_pos_arr[-1,:] = np.reshape(self.terminalModel.p_ee(TO_states[-1,:]),-1)
            TO_step_cost[-1] = self.terminalModel.cost(TO_states[-1,:], TO_controls[-1,:])
            success_flag = 1
            if abs(TO_total_cost) > 1e3:
                success_flag = 0
            #Vx_fun = opti.to_function('Vx_fun', [x0] + xs, [casadi.gradient(total_cost, x) for x in xs])
            #self.Vx_all_values = np.array(Vx_fun(ICS_state[:-1], *[opti.value(x) for x in xs])).squeeze()

            
        except:            
            print('ERROR in convergence, returning debug values')
            #print exit status
            print(opti.stats()['return_status'])
            #TO_states = np.array([[ opti.debug.value(x) for x in xs ]]).T ### DEBUG
            #TO_controls = np.array([[ opti.debug.value(u) for u in us ]]).T ### DEBUG
            TO_states = np.array([ opti.debug.value(x) for x in xs ])
            TO_controls = np.array([ opti.debug.value(u) for u in us ])
            TO_total_cost = None
            TO_ee_pos_arr = None
            TO_step_cost = None
            success_flag = 0

        return success_flag, TO_controls, TO_states, TO_ee_pos_arr, TO_total_cost, TO_step_cost
    
    def TO_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        ''' Retrieve TO problem solution and compute the value function derviative with respect to the state '''
        success_flag, TO_controls, TO_states, TO_ee_pos_arr, _, TO_step_cost = self.TO_System_Solve(ICS_state, init_TO_states, init_TO_controls, T)

        if self.w_S != 0:
            # Compute V gradient w.r.t. x (no computation dV/dt)
            dVdx = self.backward_pass(T+1, TO_states, TO_controls) 

        else:
            dVdx = np.zeros((T+1, self.conf.nb_state))

        # Add the last state component (time)
        TO_states = np.concatenate((TO_states, init_TO_states[0,-1] + np.transpose(self.conf.dt*np.array([range(T+1)]))), axis=1)
            
        return TO_states[:,:-1], TO_controls, TO_ee_pos_arr, TO_step_cost[:-1], success_flag, dVdx[:,:-1]

    def backward_pass(self, T, TO_states, TO_controls, mu=1e-9):
        ''' Perform the backward-pass of DDP to obtain the derivatives of the Value function w.r.t x '''
        n = self.conf.nb_state-1
        m = self.conf.nb_action

        X_bar = np.zeros((T, n))
        for i in range(n):
            X_bar[:,i] = [TO_states[t,i] for t in range(T)]

        U_bar = np.zeros((T-1, m))
        for i in range(m):
            U_bar[:,i] = [TO_controls[t,i] for t in range(T-1)]
 
        # The task is defined by a quadratic cost: 
        # sum_{i=0}^T 0.5 x' l_{xx,i} x + l_{x,i} x +  0.5 u' l_{uu,i} u + l_{u,i} u + x' l_{xu,i} u
        l_x  = np.zeros((T, n))
        l_xx = np.zeros((T, n, n))
        l_u  = np.zeros((T-1, m))
        l_uu = np.zeros((T-1, m, m))
        l_xu = np.zeros((T-1, n, m))
        
        # The cost-to-go is defined by a quadratic function: 0.5 x' Q_{xx,i} x + Q_{x,i} x + ...
        Q_xx = np.zeros((T-1, n, n))
        Q_x  = np.zeros((T-1, n))
        Q_uu = np.zeros((T-1, m, m))
        Q_u  = np.zeros((T-1, m))
        Q_xu = np.zeros((T-1, n, m))
        
        x = casadi.SX.sym('x',n,1)
        u = casadi.SX.sym('u',m,1)

        running_cost = self.runningSingleModel.cost(x, u)
        terminal_cost = self.terminalModel.cost(x, u)

        running_cost_xx, running_cost_x = casadi.hessian(running_cost,x)
        running_cost_uu, running_cost_u = casadi.hessian(running_cost,u)
        running_cost_xu = casadi.jacobian(casadi.jacobian(running_cost,x),u)
        terminal_cost_xx, terminal_cost_x = casadi.hessian(terminal_cost,x)

        fun_running_cost_x   = casadi.Function('fun_running_cost_x',  [x],  [running_cost_x], ['x'], ['running_cost_x'])
        fun_running_cost_xx  = casadi.Function('fun_running_cost_xx', [x],  [running_cost_xx], ['x'], ['running_cost_xx'])
        fun_running_cost_xu  = casadi.Function('fun_running_cost_xu', [x,u],[running_cost_xu], ['x','u'], ['running_cost_xu'])
        fun_running_cost_u   = casadi.Function('fun_running_cost_u',  [u],  [running_cost_u], ['u'], ['running_cost_u'])
        fun_running_cost_uu  = casadi.Function('fun_running_cost_uu', [u],  [running_cost_uu], ['u'], ['running_cost_uu'])
        fun_terminal_cost_x  = casadi.Function('fun_terminal_cost_x', [x],  [terminal_cost_x], ['x'], ['terminal_cost_x'])
        fun_terminal_cost_xx = casadi.Function('fun_terminal_cost_xx',[x],  [terminal_cost_xx], ['x'], ['terminal_cost_xx'])

        x_next_x = casadi.jacobian(self.runningSingleModel.x_next(x,u), x)
        x_next_u = casadi.jacobian(self.runningSingleModel.x_next(x,u), u)

        fun_x_next_x = casadi.Function('fun_x_next_x', [x,u], [x_next_x], ['x','u'], ['x_next_x'])
        fun_x_next_u = casadi.Function('fun_x_next_u', [x,u], [x_next_u], ['x','u'], ['x_next_u'])
        
        # The Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
        V_xx = np.zeros((T, n, n))
        V_x  = np.zeros((T, n+1))

        # Dynamics derivatives w.r.t. x and u
        A = np.zeros((T-1, n, n))
        B = np.zeros((T-1, n, m))
        
        # Initialize value function
        l_x[-1,:], l_xx[-1,:,:] = np.reshape(fun_terminal_cost_x(X_bar[-1,:]),n), fun_terminal_cost_xx(X_bar[-1,:])
        V_xx[T-1,:,:] = l_xx[-1,:,:]
        V_x[T-1,:-1]    = l_x[-1,:]

        for i in range(T-2, -1, -1):
            # Compute dynamics Jacobians
            A[i,:,:], B[i,:,:] = fun_x_next_x(X_bar[i,:], U_bar[i,:]), fun_x_next_u(X_bar[i,:], U_bar[i,:]) #self.env.augmented_derivative(X_bar[i,:], U_bar[i,:])

            # Compute the gradient of the cost function at X=X_bar
            l_x[i,:], l_xx[i,:,:] = np.reshape(fun_running_cost_x(X_bar[i,:]),n), fun_running_cost_xx(X_bar[i,:])
            l_u[i,:],l_uu[i,:,:]  = np.reshape(fun_running_cost_u(U_bar[i,:]),m), fun_running_cost_uu(U_bar[i,:])
            l_xu[i,:,:] = fun_running_cost_xu(X_bar[i,:], U_bar[i,:])                                                            
            
            # Compute regularized cost-to-go
            Q_x[i,:]     = l_x[i,:] + A[i,:,:].T @ V_x[i+1,:-1]
            Q_u[i,:]     = l_u[i,:] + B[i,:,:].T @ V_x[i+1,:-1]
            Q_xx[i,:,:]  = l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            Q_uu[i,:,:]  = l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            Q_xu[i,:,:]  = l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
                
            Qbar_uu       = Q_uu[i,:,:] + mu*np.identity(m)
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu)

            # Compute the derivative of the Value function w.r.t. x   
            V_x[i,:-1]    = Q_x[i,:]  - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_u[i,:]
            V_xx[i,:]   = Q_xx[i,:] - Q_xu[i,:,:] @ Qbar_uu_pinv @ Q_xu[i,:,:].T

        return V_x
    
import jax
import functools
import numpy as np
import jax.numpy as jnp

from jaxadi import convert
from trajax import optimizers


class TO_JAX:
    def __init__(self, env_class, conf):
        self.env = env_class('running_model', conf)
        self.conf = conf

        # JIT-compiled functions
        self.p_ee_jax = jax.jit(convert(self.env.p_ee))
        self.cost_func = jax.jit(convert(self.env.cost))
        self.dynamics_func = jax.jit(convert(self.env.x_next))
        self.dynamics_tau_func = jax.jit(convert(self.env.x_next_aug_tau))
    
    def dynamics_func_nolist(self, x, u, t):
        return jnp.squeeze(self.dynamics_func(x, u)[0])

    def cost_func_nolist(self, x, u, t, params):
        return jnp.squeeze(self.cost_func(x, u)[0]) #jnp.where(t < params, 1, 0)*
    
    def create_TO_ws(self, ICS, t_ICS, actor_model, init=1):
        @jax.jit
        def create_TO_ws_init_jax(ICS, t_ICS):
            ''' Create initial state and initial controls for TO using JAX '''        
            def single_rollout(x0):
                """Runs a single trajectory using lax.scan"""
                def dynamics_for_scan(x, _):
                    tau = jnp.zeros((self.conf.nb_action))
                    tmp = self.dynamics_tau_func(x, tau.reshape(-1, 1))
                    x_next = jnp.squeeze(tmp[0])
                    u = jnp.squeeze(tmp[1])

                    return x_next, (x_next, u) 
            
                _, (X, U) = jax.lax.scan(dynamics_for_scan, x0, jnp.arange(self.conf.NSTEPS))
                X = jnp.vstack((x0, X))

                return X, U
        
            x0 = jnp.concatenate((ICS, jnp.zeros((ICS.shape[0],1))*self.conf.dt), axis=1)
            X_batch, U_batch = jax.vmap(single_rollout)(x0)

            success_init_flag = 1

            return X_batch, U_batch, success_init_flag
        
        @jax.jit
        def create_TO_ws_jax(ICS, t_ICS):
            ''' Create initial state and initial controls for TO using JAX '''        
            def single_rollout(x0):
                """Runs a single trajectory using lax.scan"""
                def dynamics_for_scan(x, _):
                    tau = actor_model(x)
                    tmp = self.dynamics_tau_func(x, tau.reshape(-1, 1))
                    x_next = jnp.squeeze(tmp[0])
                    u = jnp.squeeze(tmp[1])

                    return x_next, (x_next, u) 
            
                _, (X, U) = jax.lax.scan(dynamics_for_scan, x0, jnp.arange(self.conf.NSTEPS))
                X = jnp.vstack((x0, X))

                return X, U
        
            x0 = jnp.concatenate((ICS, jnp.zeros((ICS.shape[0],1))*self.conf.dt), axis=1)
            X_batch, U_batch = jax.vmap(single_rollout)(x0)

            success_init_flag = 1

            return X_batch, U_batch, success_init_flag
        
        if init == 0:
            return create_TO_ws_init_jax(ICS, t_ICS)
        else:
            return create_TO_ws_jax(ICS, t_ICS)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def single_optimization(self, x0, U_init, T, maxiter, psd_delta):
        X, U, _, _, _, _, iteration, dvdx, cost = optimizers.ilqr(
            functools.partial(self.cost_func_nolist, params=T),
            self.dynamics_func_nolist,
            x0,
            U_init,
            maxiter=maxiter,
            psd_delta=psd_delta
        )

        return (
            X,
            U,
            jax.vmap(self.p_ee_jax)(X.reshape(-1, self.conf.nx, 1))[0].squeeze(), #.reshape(-1, self.conf.nx, 1)
            cost,
            jnp.logical_and(jnp.sum(cost) < 1e3, iteration < maxiter),#
            dvdx,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def optimize_batch(self, x0_batch, U_batch, T_batch, maxiter, psd_delta):
        return jax.vmap(lambda x0, U, T: self.single_optimization(x0, U, T, maxiter, psd_delta))(
            x0_batch, U_batch, T_batch
        )

    def TO_System_Solve(self, x0_batch, U_batch, T_batch, maxiter=1000, psd_delta=1e-4):
        return self.optimize_batch(x0_batch, U_batch, T_batch, maxiter, psd_delta)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def postprocess_TO_data(self, TO_controls, TO_states, TO_step_cost, TO_ee_pos_arr, NSTEPS_SH, dVdx, init_rand_t):
        if self.conf.env_RL:
            print("Not implemented yet: the simulated and real environment has to be the same")
            import sys
            sys.exit()
        state_arr = jnp.concatenate((TO_states, jnp.arange(self.conf.NSTEPS+1)[:,None]*self.conf.dt+init_rand_t), axis=1)
        ee_pos_arr = TO_ee_pos_arr
        rwrd_arr = TO_step_cost
        dVdx = jnp.concatenate((dVdx, jnp.zeros((self.conf.NSTEPS+1,1))), axis=1)

        ep_return = jnp.sum(rwrd_arr, axis=0)
        
        # Compute final_lookahead_step and done_arr
        steps = jnp.arange(self.conf.NSTEPS + 1)
        if self.conf.MC:
            final_lookahead_step = jnp.full(self.conf.NSTEPS + 1, NSTEPS_SH)
            done_arr = jnp.ones(self.conf.NSTEPS + 1)
        else:
            final_lookahead_step = jnp.minimum(steps + self.conf.nsteps_TD_N, NSTEPS_SH)
            done_arr = (steps + self.conf.nsteps_TD_N) >= NSTEPS_SH

        state_next_rollout_arr = state_arr[(final_lookahead_step + 1).astype(jnp.int32)]
        
        def compute_cost_to_go(costs: jnp.ndarray, horizon: int = self.conf.nsteps_TD_N):
            cumsum_costs = jnp.cumsum(costs)
        
            cost_to_go = cumsum_costs[horizon:] - jnp.concatenate([jnp.zeros(1), cumsum_costs[:-horizon-1]])
            last_sums = jnp.flip(jnp.cumsum(jnp.flip(costs[-horizon:]))) 

            return jnp.concatenate([cost_to_go, last_sums])

        partial_reward_to_go_arr = compute_cost_to_go(rwrd_arr, self.conf.nsteps_TD_N)

        return state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ep_return, ee_pos_arr, dVdx
