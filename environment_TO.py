import jax
import casadi
import numpy as np
import jax.numpy as jnp
import pinocchio.casadi as cpin
from jaxadi import convert

def wrap_convert(f):
    """Convert a CasADi function into a JAX-compatible and JIT-compiled function."""
    f_converted = convert(f)
    f_jitted = jax.jit(f_converted)

    @jax.jit
    def wrapped(*args):
        # Squeeze the first element (CasADi returns a tuple)
        return jnp.squeeze(f_jitted(*args)[0])

    return wrapped

class SingleIntegrator_CAMS:
    def __init__(self, name, conf):
        """
        name : (str) Name of the casadi-model (either 'running' or 'terminal')
        conf : (object) Configuration object containing the following attributes:
            robot :                      (RobotWrapper instance).
            cmodel :                     (Casadi-Pinocchio instance).
            cdata :                      (Casadi-Pinocchio model data).                   
            dt :                         (float) Timestep.
            end_effector_frame_id :      (str) Name of the end effector frame.

            # Cost function parameters
            TARGET_STATE :               (float array) Target position.
            cost_funct_param             (float array) Cost function scale and offset factors.
            soft_max_param :             (float array) Soft parameters array.
            obs_param :                  (float array) Obtacle parameters array.
            cost_weights_running :       (float array) Running cost weights vector.
            cost_weights_terminal :      (float array) Terminal cost weights vector.
            nb_action :                  (int) Dimensionality of the action vector.
            u_max :                      (float array) Maximum magnitude of the action output.
        """

        self.name = name
        self.conf = conf
        
        self.nx = self.conf.nx
        self.nu = self.conf.na

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

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        cxt = casadi.SX.sym("x",self.nx+1,1)
        cu = casadi.SX.sym("u",self.nu,1)

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()
        
        self.p_ee = casadi.Function('p_ee', [cxt], [self.get_end_effector_position_fun(cxt)])
        
        self.x_next_tau = casadi.Function('x_next', [cxt, cu], [self.simulate_fun_tau(cxt,cu)])
        self.cost_tau = casadi.Function('cost', [cxt,cu], [self.cost_fun_tau(cxt,cu)])
        self.check_feasible = casadi.Function('check_ICS_feasible', [cxt], [self.check_ICS_feasible(cxt)])

        self.p_ee_jax_wrapped = wrap_convert(self.p_ee)
        self.cost_tau_func_wrapped = wrap_convert(self.cost_tau)   
        self.dynamics_tau_func_wrapped = wrap_convert(self.x_next_tau)
            
        self.check_feasible_wrapped = wrap_convert(self.check_feasible)    

    def get_end_effector_position_fun(self, x): 
        """Compute end effector position."""     

        return casadi.vertcat(x[:2],0)
    
    def bound_control_cost(self, action):
        """Compute cost c(u)."""
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**8

        return u_cost
    
    def cost_fun_tau(self, x, u):
        """Compute cost c(x,u)."""
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (casadi.log(casadi.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (casadi.log(casadi.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (casadi.log(casadi.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Control effort term ###
        u_cost = self.bound_control_cost(u)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = casadi.log(casadi.exp(self.alpha2*-(casadi.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + casadi.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*casadi.sqrt(0.1))) + 1)/self.alpha2
        
        ### Terminal cost on final velocity ###
        v_cost = 0
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost + self.weights[6]*u_cost - self.offset)
 
        return cost
    
    def simulate_fun_tau(self, x, u):
        """Integrate dynamics."""
        a = u
        x_next = x + self.conf.dt*casadi.vertcat(a, 1)
        
        return x_next
    
    def check_ICS_feasible(self, x):
        """Check if ICS is feasible."""
        p_ee = self.p_ee(x)
        
        ellipses = casadi.vertcat(
        ((p_ee[0] - self.conf.XC1) ** 2) / ((self.conf.A1 / 2) ** 2) + ((p_ee[1] - self.conf.YC1) ** 2) / ((self.conf.B1 / 2) ** 2),
        ((p_ee[0] - self.conf.XC2) ** 2) / ((self.conf.A2 / 2) ** 2) + ((p_ee[1] - self.conf.YC2) ** 2) / ((self.conf.B2 / 2) ** 2),
        ((p_ee[0] - self.conf.XC3) ** 2) / ((self.conf.A3 / 2) ** 2) + ((p_ee[1] - self.conf.YC3) ** 2) / ((self.conf.B3 / 2) ** 2),
        )

        feasible_flags = casadi.mmin(ellipses) > 1

        return feasible_flags  

class DoubleIntegrator_CAMS:
    def __init__(self, name, conf):
        """
        name : (str) Name of the casadi-model (either 'running' or 'terminal')
        conf : (object) Configuration object containing the following attributes:
            robot :                      (RobotWrapper instance).
            cmodel :                     (Casadi-Pinocchio instance).
            cdata :                      (Casadi-Pinocchio model data).                   
            dt :                         (float) Timestep.
            end_effector_frame_id :      (str) Name of the end effector frame.

            # Cost function parameters
            TARGET_STATE :               (float array) Target position.
            cost_funct_param             (float array) Cost function scale and offset factors.
            soft_max_param :             (float array) Soft parameters array.
            obs_param :                  (float array) Obtacle parameters array.
            cost_weights_running :       (float array) Running cost weights vector.
            cost_weights_terminal :      (float array) Terminal cost weights vector.
            nb_action :                  (int) Dimensionality of the action vector.
            u_max :                      (float array) Maximum magnitude of the action output.
        """

        self.name = name
        self.conf = conf
        
        self.nq = self.conf.cmodel.nq
        self.nv = self.conf.cmodel.nv
        self.nx = self.nq+self.nv
        self.nu = self.conf.cmodel.nv

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

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        cxt = casadi.SX.sym("x",self.conf.nb_state,1)
        cu = casadi.SX.sym("u",self.conf.nb_action,1)


        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cxt[:self.nq])
        self.rnea = casadi.Function('rnea', [cxt, cu], [cpin.rnea(self.conf.cmodel, self.conf.cdata, cxt[:self.conf.nq], cxt[self.conf.nq:-1], cu)])
        self.p_ee = casadi.Function('p_ee', [cxt], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation]) #self.get_end_effector_position_fun(cx)])#
        self.cost = casadi.Function('cost', [cxt,cu], [self.cost_fun(cxt, cu)])
        self.cost_tau = casadi.Function('cost', [cxt,cu], [self.cost_fun_tau(cxt,cu)])
        self.x_next = casadi.Function('x_next', [cxt, cu], [self.simulate_fun(cxt,cu)])
        self.x_next_tau = casadi.Function('aba', [cxt, cu], [self.simulate_fun_tau(cxt,cu)])
        self.get_u_from_tau = casadi.Function('get_u_from_tau', [cxt, cu], [self.get_u_from_tau_fun(cxt,cu)])
        self.check_feasible = casadi.Function('check_ICS_feasible', [cxt], [self.check_ICS_feasible(cxt)])        
        
        self.p_ee_jax_wrapped = wrap_convert(self.p_ee)
        self.cost_func_wrapped = wrap_convert(self.cost)
        self.cost_tau_func_wrapped = wrap_convert(self.cost_tau)   
        self.dynamics_func_wrapped = wrap_convert(self.x_next)
        self.dynamics_tau_func_wrapped = wrap_convert(self.x_next_tau)
        self.get_u_from_tau_func_wrapped = wrap_convert(self.get_u_from_tau)
            
        self.check_feasible_wrapped = wrap_convert(self.check_feasible)    

    def bound_control_cost(self, action):
        """Compute cost c(u)."""
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**8

        return u_cost
    
    def cost_fun_x(self, x):
        """Compute cost c(x)."""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])
        p_ee = self.p_ee(x)

        ### Penalties representing the obstacle ###
        ell1_cost = (casadi.log(casadi.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (casadi.log(casadi.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (casadi.log(casadi.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = casadi.log(casadi.exp(self.alpha2*-(casadi.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + casadi.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*casadi.sqrt(0.1))) + 1)/self.alpha2
        
        ### Terminal cost on final velocity ###
        v_cost = 0
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost - self.offset)
 
        return cost

    def cost_fun(self, x, u):
        """Compute cost c(x,u) (inverse dynamics)."""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])
    
        ### Control effort term ###
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(self.rnea(x,u)))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def cost_fun_tau(self, x, u):
        """Compute cost"""
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(u))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def simulate_fun(self, x, u):
        """Integrate dynamics (inverse dynamics)."""
        a = u
        x_next = x + self.conf.dt*casadi.vertcat(x[self.nq:-1], a, 1)
        
        return x_next

    def get_u_from_tau_fun(self, x, u):
        """Compute accelerations from torques"""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata, x[:self.nq])
        M = cpin.crba(self.conf.cmodel, self.conf.cdata, x[:self.nq])
        tau_rnea = cpin.rnea(self.conf.cmodel, self.conf.cdata, x[:self.nq], x[self.nq:-1], casadi.SX.zeros(self.nq))    
        a = casadi.solve(M, u - tau_rnea)
        
        return a
    
    def simulate_fun_tau(self, x, u):
        """Integrate dynamics."""
        a = self.get_u_from_tau_fun(x, u)
        
        x_next = x + self.conf.dt*casadi.vertcat(x[self.nq:-1], a, 1)
        
        return x_next
    
    def check_ICS_feasible(self, x):
        """Check if ICS is feasible"""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])
        p_ee = self.p_ee(x)
        
        ellipses = casadi.vertcat(
        ((p_ee[0] - self.conf.XC1) ** 2) / ((self.conf.A1 / 2) ** 2) + ((p_ee[1] - self.conf.YC1) ** 2) / ((self.conf.B1 / 2) ** 2),
        ((p_ee[0] - self.conf.XC2) ** 2) / ((self.conf.A2 / 2) ** 2) + ((p_ee[1] - self.conf.YC2) ** 2) / ((self.conf.B2 / 2) ** 2),
        ((p_ee[0] - self.conf.XC3) ** 2) / ((self.conf.A3 / 2) ** 2) + ((p_ee[1] - self.conf.YC3) ** 2) / ((self.conf.B3 / 2) ** 2),
        )

        feasible_flags = casadi.mmin(ellipses) > 1

        return feasible_flags  

class Reacher_CAMS:
    def __init__(self, name, conf):
        """
        name : (str) Name of the casadi-model (either 'running' or 'terminal')
        conf : (object) Configuration object containing the following attributes:
            robot :                      (RobotWrapper instance).
            cmodel :                     (Casadi-Pinocchio instance).
            cdata :                      (Casadi-Pinocchio model data).                   
            dt :                         (float) Timestep.
            end_effector_frame_id :      (str) Name of the end effector frame.

            # Cost function parameters
            cost_funct_param             (float array) Cost function scale and offset factors.
            cost_weights_running :       (float array) Running cost weights vector.
            cost_weights_terminal :      (float array) Terminal cost weights vector.
            nb_action :                  (int) Dimensionality of the action vector.
            u_max :                      (float array) Maximum magnitude of the action output.
        """

        self.name = name
        self.conf = conf
        
        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]
        
        cxt = casadi.SX.sym("x",self.conf.nb_state,1)
        cu = casadi.SX.sym("u",self.conf.nb_action,1)
        p = casadi.SX.sym("u",2,1)

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cxt[:2])
        self.rnea = casadi.Function('rnea', [cxt, cu], [cpin.rnea(self.conf.cmodel, self.conf.cdata, cxt[:2], cxt[2:4], cu)])
        self.p_ee = casadi.Function('p_ee', [cxt], [self.get_end_effector_position(cxt)])
                
        self.x_next = casadi.Function('x_next', [cxt, cu], [self.simulate_fun(cxt,cu)])
        self.x_next_tau = casadi.Function('aba', [cxt, cu], [self.simulate_fun_tau(cxt,cu)])
        self.cost = casadi.Function('cost', [cxt,cu], [self.cost_fun(cxt,cu)])
        self.cost_tau = casadi.Function('cost', [cxt,cu], [self.cost_fun_tau(cxt,cu)])
        self.get_u_from_tau = casadi.Function('get_u_from_tau', [cxt, cu], [self.get_u_from_tau_fun(cxt,cu)])
        self.check_feasible = casadi.Function('check_ICS_feasible', [cxt], [self.check_ICS_feasible(cxt)])

        self.p_ee_jax_wrapped = wrap_convert(self.p_ee)
        self.cost_func_wrapped = wrap_convert(self.cost)
        self.cost_tau_func_wrapped = wrap_convert(self.cost_tau)       
        self.dynamics_func_wrapped = wrap_convert(self.x_next)
        self.dynamics_tau_func_wrapped = wrap_convert(self.x_next_tau)
        self.get_u_from_tau_func_wrapped = wrap_convert(self.get_u_from_tau)
        self.check_feasible_wrapped = wrap_convert(self.check_feasible)  

    def get_end_effector_position(self, x):
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.conf.nq])
        return self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation

    def simulate_fun(self, x, u):
        """Integrate dynamics."""
        a = u

        x_next = casadi.SX(self.conf.nb_state,1)
        x_next[0] = x[0] + self.conf.dt*x[2]
        x_next[1] = x[1] + self.conf.dt*x[3]
        x_next[2] = x[2] + self.conf.dt*a[0]
        x_next[3] = x[3] + self.conf.dt*a[1]
        x_next[4] = x[4]
        x_next[5] = x[5]
        x_next[-1] = x[-1] + self.conf.dt

        distance_tip_target = self.p_ee(x_next)[:2]-x[4:6]
        x_next[6:8] = distance_tip_target

        x_next = x + self.conf.dt*casadi.vertcat(x[self.conf.nq:self.conf.nx], a, 1e-16*x[4:6], (-x[6:8]+distance_tip_target)/self.conf.dt,1)

        return x_next

    def get_u_from_tau_fun(self, x, u):
        """Integrate dynamics"""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata, x[:2])
        M = cpin.crba(self.conf.cmodel, self.conf.cdata, x[:2])
        tau_rnea = cpin.rnea(self.conf.cmodel, self.conf.cdata, x[:2], x[2:4], casadi.SX.zeros(2))    
        a = casadi.solve(M, u - tau_rnea)
        
        return a
    
    def simulate_fun_tau(self, x, u):
        """Integrate dynamics"""
        a = self.get_u_from_tau_fun(x,u)
        
        x_next = casadi.SX(self.conf.nb_state,1)
        x_next[0] = x[0] + self.conf.dt*x[2]
        x_next[1] = x[1] + self.conf.dt*x[3]
        x_next[2] = x[2] + self.conf.dt*a[0]
        x_next[3] = x[3] + self.conf.dt*a[1]
        x_next[4] = x[4]
        x_next[5] = x[5]
        x_next[-1] = x[-1] + self.conf.dt

        distance_tip_target = self.p_ee(x_next)[:2]-x_next[4:6]
        x_next[6:8] = distance_tip_target

        x_next = x + self.conf.dt*casadi.vertcat(x[self.conf.nq:self.conf.nx], a, 1e-16*x[4:6], (-x[6:8]+distance_tip_target)/self.conf.dt,1)
        
        return x_next
    
    def bound_control_cost(self, action):
        """Compute cost c(u)."""
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i]/(15*self.scale*self.weights[6])
        return u_cost
    
    def cost_fun_x(self, x): 
        """Compute cost c(x)."""
        cost = casadi.sum1((x[6:8])**2)
        
        return cost
    
    def cost_fun(self, x, u):
        """Compute cost c(x,u) (inverse dynamics)."""
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(self.rnea(x,u)))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def cost_fun_tau(self, x, u):
        """Compute cost c(x,u)."""
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(u))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def check_ICS_feasible(self, x):
        """Check if ICS is feasible"""

        return 1    
    
class Car_CAMS:
    def __init__(self, name, conf):
        """
        name : (str) Name of the casadi-model (either 'running' or 'terminal')
        conf : (object) Configuration object containing the following attributes:           
            dt :                         (float) Timestep.

            # Cost function parameters
            TARGET_STATE :               (float array) Target position.
            cost_funct_param             (float array) Cost function scale and offset factors.
            soft_max_param :             (float array) Soft parameters array.
            obs_param :                  (float array) Obtacle parameters array.
            cost_weights_running :       (float array) Running cost weights vector.
            cost_weights_terminal :      (float array) Terminal cost weights vector.
            nb_action :                  (int) Dimensionality of the action vector.
            u_max :                      (float array) Maximum magnitude of the action output.
        """

        self.name = name
        self.conf = conf
        
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

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        cxt = casadi.SX.sym("x",self.conf.nb_state,1)
        cu = casadi.SX.sym("u",self.conf.nb_action,1)

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        self.p_ee = casadi.Function('p_ee', [cxt], [self.get_end_effector_position_fun(cxt)])
        
        self.x_next_tau = casadi.Function('aba', [cxt, cu], [self.simulate_fun(cxt,cu)])
        self.cost_tau = casadi.Function('cost', [cxt,cu], [self.cost_fun(cxt,cu)])
        self.check_feasible = casadi.Function('check_ICS_feasible', [cxt], [self.check_ICS_feasible(cxt)])

        self.p_ee_jax_wrapped = wrap_convert(self.p_ee)
        self.cost_tau_func_wrapped = wrap_convert(self.cost_tau)       
        self.dynamics_tau_func_wrapped = wrap_convert(self.x_next_tau)
        self.check_feasible_wrapped = wrap_convert(self.check_feasible)  

    def get_end_effector_position_fun(self, x):        
        """Compute end effector position."""
        return casadi.vertcat(x[:2], 0)
    
    def bound_control_cost(self, action):
        """Compute cost c(u)."""
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**8

        return u_cost
    
    def cost_fun_x(self, x):
        """Compute cost c(x)."""
        p_ee = self.p_ee(x)
        x = x[:-1]
        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost - self.offset)
 
        return cost
    
    def cost_fun(self, x, u):
        """Compute cost c(x,u)."""
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(u))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def simulate_fun(self, x, u):
        """Integrate dynamics."""
        dt2_over_2 = (self.conf.dt*self.conf.dt)/2

        c = x[2]
        s = x[3]

        x0 = x[0] + self.conf.dt*x[4]*c + dt2_over_2*x[5]*c
        x1 = x[1] + self.conf.dt*x[4]*s + dt2_over_2*x[5]*s
        x2 = x[2] - self.conf.dt * s * u[0]
        x3 = x[3] + self.conf.dt * c * u[0]
        norm = casadi.sqrt(x2**2 + x3**2)
        x2 /= norm
        x3 /= norm
        x4 = x[4] + self.conf.dt*x[5]
        x5 = x[5] + self.conf.dt*u[1]
        x6 = x[6] + self.conf.dt

        x_next = casadi.vertcat(x0, x1, x2, x3, x4, x5, x6)

        return x_next

    def check_ICS_feasible(self, x):
        """Check if ICS is feasible"""
        p_ee = self.p_ee(x)
        
        ellipses = casadi.vertcat(
        ((p_ee[0] - self.conf.XC1) ** 2) / ((self.conf.A1 / 2) ** 2) + ((p_ee[1] - self.conf.YC1) ** 2) / ((self.conf.B1 / 2) ** 2),
        ((p_ee[0] - self.conf.XC2) ** 2) / ((self.conf.A2 / 2) ** 2) + ((p_ee[1] - self.conf.YC2) ** 2) / ((self.conf.B2 / 2) ** 2),
        ((p_ee[0] - self.conf.XC3) ** 2) / ((self.conf.A3 / 2) ** 2) + ((p_ee[1] - self.conf.YC3) ** 2) / ((self.conf.B3 / 2) ** 2),
        )

        feasible_flags = casadi.mmin(ellipses) > 1

        return feasible_flags  

class Manipulator_CAMS:
    def __init__(self, name, conf):
        """
        name : (str) Name of the casadi-model (either 'running' or 'terminal')
        conf : (object) Configuration object containing the following attributes:
            robot :                      (RobotWrapper instance).
            cmodel :                     (Casadi-Pinocchio instance).
            cdata :                      (Casadi-Pinocchio model data).                   
            dt :                         (float) Timestep.
            end_effector_frame_id :      (str) Name of the end effector frame.

            # Cost function parameters
            TARGET_STATE :               (float array) Target position.
            cost_funct_param             (float array) Cost function scale and offset factors.
            soft_max_param :             (float array) Soft parameters array.
            obs_param :                  (float array) Obtacle parameters array.
            cost_weights_running :       (float array) Running cost weights vector.
            cost_weights_terminal :      (float array) Terminal cost weights vector.
            nb_action :                  (int) Dimensionality of the action vector.
            u_max :                      (float array) Maximum magnitude of the action output.
        """

        self.name = name
        self.conf = conf
        
        self.nq = self.conf.cmodel.nq
        self.nv = self.conf.cmodel.nv
        self.nx = self.nq+self.nv
        self.nu = self.conf.cmodel.nv

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

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        cxt = casadi.SX.sym("x",self.conf.nb_state,1)
        cu = casadi.SX.sym("u",self.conf.nb_action,1)

        if self.name == 'running_model':
            self.weights = np.copy(self.conf.cost_weights_running)
        elif self.name == 'terminal_model':
            self.weights = np.copy(self.conf.cost_weights_terminal)
        else:
            print("The model can be either 'running_model' or 'terminal_model'")
            import sys
            sys.exit()

        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,cxt[:self.nq])
        self.rnea = casadi.Function('rnea', [cxt, cu], [cpin.rnea(self.conf.cmodel, self.conf.cdata, cxt[:self.conf.nq], cxt[self.conf.nq:-1], cu)])
        self.p_ee = casadi.Function('p_ee', [cxt], [self.conf.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation]) #self.get_end_effector_position_fun(cx)])#
        self.cost = casadi.Function('cost', [cxt,cu], [self.cost_fun(cxt, cu)])
        self.cost_tau = casadi.Function('cost', [cxt,cu], [self.cost_fun_tau(cxt,cu)])
        self.x_next = casadi.Function('x_next', [cxt, cu], [self.simulate_fun(cxt,cu)])
        self.get_u_from_tau = casadi.Function('get_u_from_tau', [cxt, cu], [self.get_u_from_tau_fun(cxt,cu)])
        self.x_next_tau = casadi.Function('aba', [cxt, cu], [self.simulate_fun_tau(cxt,cu)])
        self.check_feasible = casadi.Function('check_ICS_feasible', [cxt], [self.check_ICS_feasible(cxt)])

        self.p_ee_jax = jax.jit(convert(self.p_ee))

        self.p_ee_jax_wrapped = wrap_convert(self.p_ee)
        self.cost_func_wrapped = wrap_convert(self.cost)
        self.cost_tau_func_wrapped = wrap_convert(self.cost_tau)       
        self.dynamics_func_wrapped = wrap_convert(self.x_next)
        self.dynamics_tau_func_wrapped = wrap_convert(self.x_next_tau)
        self.get_u_from_tau_func_wrapped = wrap_convert(self.get_u_from_tau)
        self.check_feasible_wrapped = wrap_convert(self.check_feasible)    

    def bound_control_cost(self, action):
        """Compute cost c(u)."""
        u_cost = 0
        for i in range(self.conf.nb_action):
            u_cost += action[i]*action[i] + self.conf.w_b*(action[i]/self.conf.u_max[i])**6

        return u_cost
    
    def cost_fun_x(self, x):
        """Compute cost c(x)."""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])

        p_ee = self.p_ee(x)
        x = x[:-1]
        ### Penalties representing the obstacle ###
        ell1_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC1)**2)/((self.A1/2)**2) + ((p_ee[1]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha)  
        ell2_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC2)**2)/((self.A2/2)**2) + ((p_ee[1]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha) 
        ell3_cost = (np.log(np.exp(self.alpha*-(((p_ee[0]-self.XC3)**2)/((self.A3/2)**2) + ((p_ee[1]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha)

        ### Distence to target term (quadratic term) ###
        dist_cost = (p_ee[0]-self.x_des)**2 + (p_ee[1]-self.y_des)**2

        ### Distence to target term (log valley centered at target) ###      
        peak_rew = np.log(np.exp(self.alpha2*-(np.sqrt((p_ee[0]-self.x_des)**2 +0.1) - 0.1 + np.sqrt((p_ee[1]-self.y_des)**2 +0.1) - 0.1 -2*np.sqrt(0.1))) + 1)/self.alpha2
        
        ### Terminal cost on final velocity ###
        v_cost = 0
        
        cost = self.scale*(self.weights[0]*dist_cost - self.weights[1]*peak_rew + self.weights[2]*v_cost + self.weights[3]*ell1_cost + self.weights[4]*ell2_cost + self.weights[5]*ell3_cost - self.offset)
 
        return cost
    
    def cost_fun(self, x, u):
        """Compute cost c(x,u) (inverse dynamics)."""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])
    
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(self.rnea(x,u)))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def cost_fun_tau(self, x, u):
        """Compute cost."""
        u_cost = self.scale*(self.weights[6]*self.bound_control_cost(u))
        x_cost = self.cost_fun_x(x)
        cost = u_cost + x_cost
 
        return cost
    
    def simulate_fun(self, x, u):
        """Integrate dynamics (inverse dynamics)."""
        a = u

        x_next = x + self.conf.dt*casadi.vertcat(x[self.nq:-1], a, 1)
        
        return x_next
    
    def get_u_from_tau_fun(self, x, u):
        """Compute acceleration given tau."""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])
        M = cpin.crba(self.conf.cmodel, self.conf.cdata, x[:self.nq])
        tau_rnea = cpin.rnea(self.conf.cmodel, self.conf.cdata, x[:self.nq], x[self.nq:-1], casadi.SX.zeros(self.nq))    
        a = casadi.solve(M, u - tau_rnea)
        
        return a
    
    def simulate_fun_tau(self, x, u):
        """Integrate dynamics."""
        a = self.get_u_from_tau_fun(x, u)

        x_next = x + self.conf.dt*casadi.vertcat(x[self.nq:-1], a, 1)
        
        return x_next
    
    def check_ICS_feasible(self, x):
        """Check if ICS is feasible."""
        cpin.framesForwardKinematics(self.conf.cmodel, self.conf.cdata,x[:self.nq])
        p_ee = self.p_ee(x)
        
        ellipses = casadi.vertcat(
        ((p_ee[0] - self.conf.XC1) ** 2) / ((self.conf.A1 / 2) ** 2) + ((p_ee[1] - self.conf.YC1) ** 2) / ((self.conf.B1 / 2) ** 2),
        ((p_ee[0] - self.conf.XC2) ** 2) / ((self.conf.A2 / 2) ** 2) + ((p_ee[1] - self.conf.YC2) ** 2) / ((self.conf.B2 / 2) ** 2),
        ((p_ee[0] - self.conf.XC3) ** 2) / ((self.conf.A3 / 2) ** 2) + ((p_ee[1] - self.conf.YC3) ** 2) / ((self.conf.B3 / 2) ** 2),
        )

        feasible_flags = casadi.mmin(ellipses) > 1

        return feasible_flags    

class AlienGo_CAMS:
    def __init__(self, name, conf):
        '''
        :name :                                 (str) Name of the casadi-model (either 'running' or 'terminal')

        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param cmodel :                     (Casadi-Pinocchio instance)
            :param cdata :                      (Casadi-Pinocchio model data)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
            :param cost_weights_running :       (float array) Running cost weights vector
            :param cost_weights_terminal :      (float array) Terminal cost weights vector
        '''
        self.name = name
        self.conf = conf

        self.x_des = self.conf.TARGET_STATE[0]
        self.y_des = self.conf.TARGET_STATE[1]

        cxt = casadi.SX.sym("x",self.conf.nb_state,1)
        cu = casadi.SX.sym("u",self.conf.nb_action,1)

        self.p_ee = casadi.Function('p_ee', [cxt], [self.get_end_effector_position_fun(cxt)])
        self.cost_tau = casadi.Function('cost', [cxt,cu], [self.cost_fun(cxt, cu)])
        self.x_next_tau = casadi.Function('x_next', [cxt, cu], [self.simulate_fun(cxt,cu)])
        self.check_feasible = casadi.Function('check_ICS_feasible', [cxt], [self.check_ICS_feasible(cxt)])
        
        self.p_ee_jax_wrapped = wrap_convert(self.p_ee)
        self.cost_tau_func_wrapped = wrap_convert(self.cost_tau)
        self.dynamics_tau_func_wrapped = wrap_convert(self.x_next_tau)
        self.check_feasible_wrapped = wrap_convert(self.check_feasible)    

    def get_end_effector_position_fun(self, x):
        ''' Compute end effector position '''
        p_ee = casadi.SX(3,1)
        p_ee[0] = x[4]
        p_ee[1] = x[5]
        p_ee[2] = 0
        
        return p_ee

    def bound_control_cost(self, action):
        """Compute cost c(u)."""
        running_cost_p  = self.conf.wp*casadi.sumsqr(action[:4])  
        running_cost_a  = self.conf.wa*(action[4]**2)            
        running_cost_dt = self.conf.wdt*(action[5]**2)           
        u_cost = running_cost_p + running_cost_a + running_cost_dt
        for i in range(action.shape[0]):
            u_cost += self.conf.w_b*(action[i]/(self.conf.u_max[i]*0.9))**8

        return u_cost
    
    def cost_fun(self, x, u):
        """Compute cost."""
        delta_p_next, a, dt = u[:4], (u[4] + 0.5), (u[5] + self.conf.dt_delta)

        delta_p = x[:4]
        x_CoM = x[4:6]
        dx_CoM = x[6:8]
        obstacle_state = x[8:10]
        wall_state = x[10:14]

        running_cost_c = self.conf.wc*2*casadi.log(casadi.sumsqr(x_CoM-self.conf.TARGET_STATE)+1)
        running_cost_dc = self.conf.wdc*casadi.sumsqr(dx_CoM)

        peak_rew = self.conf.w_peak*casadi.log(casadi.exp(self.conf.alpha_peak*-(casadi.sqrt((x_CoM[0]-self.conf.TARGET_STATE[0])**2 +0.1) - 0.1 + casadi.sqrt((x_CoM[1]--self.conf.TARGET_STATE[1])**2 +0.1) - 0.1 -2*casadi.sqrt(0.1))) + 1)/self.conf.alpha_peak

        running_dx_CoM_bound_cost = self.conf.w_b*((dx_CoM[0]/self.conf.vx_max)**8+(dx_CoM[1]/self.conf.vy_max)**8)

        # Cost on feet position distance with respect to hips
        R = casadi.SX_eye(2)

        hip_pos_0 = x_CoM + casadi.mtimes(R, self.conf.hip_0)
        hip_pos_1 = x_CoM + casadi.mtimes(R, self.conf.hip_1)
        hip_pos_2 = x_CoM + casadi.mtimes(R, self.conf.hip_2)
        hip_pos_3 = x_CoM + casadi.mtimes(R, self.conf.hip_3)
        hip_pos_list = [hip_pos_0, hip_pos_1, hip_pos_2, hip_pos_3]

        hip_ps_mid_0 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_0)
        hip_ps_mid_1 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_1)
        hip_ps_mid_2 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_2)
        hip_ps_mid_3 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_3)
        hip_ps_mid_list = [hip_ps_mid_0, hip_ps_mid_1, hip_ps_mid_2, hip_ps_mid_3]
        check_pos = hip_pos_list + hip_ps_mid_list

        running_cost_obs = 0
        for i in range(4):
            running_cost_obs += self.conf.w_obs*(casadi.log(casadi.exp(self.conf.alpha_obs*- (((check_pos[i][0]-obstacle_state[0])**4)/((self.conf.A5/2)**4) + ((check_pos[i][1]-obstacle_state[1])**4)/((self.conf.B5/2)**4) - 1.0)) + 1)/self.conf.alpha_obs) 
            running_cost_obs += self.conf.w_wall*(casadi.log(casadi.exp(self.conf.alpha_obs*-(((check_pos[i][0]-                  wall_state[0])**4)/((                 self.conf.A1/2)**4) + ((check_pos[i][1]-(wall_state[1]+wall_state[3])/2)**4)/(((wall_state[1]-wall_state[3])/2)**4) - 1.0)) + 1)/self.conf.alpha_obs)
            running_cost_obs += self.conf.w_wall*(casadi.log(casadi.exp(self.conf.alpha_obs*-(((check_pos[i][0]-(wall_state[0]+wall_state[2])/2)**4)/(((wall_state[0]-wall_state[2])/2)**4) + ((check_pos[i][1]-                  wall_state[1])**4)/((                 self.conf.B2/2)**4) - 1.0)) + 1)/self.conf.alpha_obs)
            running_cost_obs += self.conf.w_wall*(casadi.log(casadi.exp(self.conf.alpha_obs*-(((check_pos[i][0]-                  wall_state[2])**4)/((                 self.conf.A3/2)**4) + ((check_pos[i][1]-(wall_state[1]+wall_state[3])/2)**4)/(((wall_state[1]-wall_state[3])/2)**4) - 1.0)) + 1)/self.conf.alpha_obs)
            running_cost_obs += self.conf.w_wall*(casadi.log(casadi.exp(self.conf.alpha_obs*-(((check_pos[i][0]-(wall_state[0]+wall_state[2])/2)**4)/(((wall_state[0]-wall_state[2])/2)**4) + ((check_pos[i][1]-                  wall_state[3])**4)/((                 self.conf.B4/2)**4) - 1.0)) + 1)/self.conf.alpha_obs)

        u_cost = self.bound_control_cost(u)

        xy_foot_0, xy_foot_1 = self.get_foot_positions(x)
        cop = xy_foot_0 + a * (xy_foot_1 - xy_foot_0)
        ddx_CoM = self.conf.w**2 * (x_CoM - cop)

        cost = running_cost_c + running_cost_dc  + running_dx_CoM_bound_cost - peak_rew + running_cost_obs + u_cost + (ddx_CoM[0]**2 + ddx_CoM[1]**2)

        return self.conf.scale_cost_fun*cost
    
    def get_foot_positions(self, x):
        """Compute feet positions."""
        delta_p = x[:4]
        x_CoM = x[4:6]

        gait_idx = (1 - casadi.cos(casadi.pi * x[-1] / 1)) / 2
        gait_0 = casadi.SX([self.conf.lx_hip, -self.conf.ly_hip])*gait_idx + casadi.SX([ self.conf.lx_hip,  self.conf.ly_hip])*(1-gait_idx)
        gait_1 = casadi.SX([-self.conf.lx_hip, self.conf.ly_hip])*gait_idx + casadi.SX([-self.conf.lx_hip, -self.conf.ly_hip])*(1-gait_idx)
        
        # Cost on feet position distance with respect to hips
        R = casadi.SX_eye(2)

        # Apply rotation to gait offsets
        rotated_gait_0 = R @ (gait_0 - delta_p[:2])
        rotated_gait_1 = R @ (gait_1 - delta_p[2:])

        # Compute hip positions
        hip_pos_0 = x_CoM + rotated_gait_0
        hip_pos_1 = x_CoM + rotated_gait_1

        # Compute the foot positions
        xy_foot_0 = hip_pos_0 
        xy_foot_1 = hip_pos_1 

        return xy_foot_0, xy_foot_1
    
    def simulate_fun(self, x, u):
        """Integrate dynamics."""
        delta_p_next, a, dt = u[:4], (u[4] + 0.5), (u[5] + self.conf.dt_delta)

        delta_p = x[:4]
        x_CoM = x[4:6]
        dx_CoM = x[6:8]
        obstacle_state = x[8:10]
        wall_state = x[10:14]

        xy_foot_0, xy_foot_1 = self.get_foot_positions(x)

        R = casadi.SX_eye(2)

        dx_CoM = R @ dx_CoM

        cop = xy_foot_0 + a * (xy_foot_1 - xy_foot_0)

        # Handle cosh and sinh for both single and batched inputs
        ch, sh = casadi.cosh(self.conf.w * dt), casadi.sinh(self.conf.w * dt)

        x_CoM_next = ch * x_CoM + sh * dx_CoM / self.conf.w + (1 - ch) * cop

        # Compute the next state
        fix_dim = casadi.SX([1, 1])*1e-8
        x_next = casadi.vertcat(
            fix_dim + delta_p_next[:2],
            fix_dim + delta_p_next[2:],
            ch * x_CoM + sh * dx_CoM / self.conf.w + (1 - ch) * cop,
            self.conf.w*sh*x_CoM + ch * dx_CoM - self.conf.w * sh * cop,
            fix_dim + obstacle_state,
            fix_dim + wall_state[:2],
            fix_dim + wall_state[2:],
            ((x_CoM_next[0]-obstacle_state[0])**2 + (x_CoM_next[1]-obstacle_state[1])**2)**0.5,
            ((x_CoM_next[0])**2 + (x_CoM_next[1])**2)**0.5,
            x[-1] + 1.000
        )

        return x_next
    
    def check_ICS_feasible(self, x):
        """Check if ICS is feasible."""
        p_ee = self.p_ee(x)

        delta_p = x[:4]
        x_CoM = x[4:6]
        dx_CoM = x[6:8]
        obstacle_state = x[8:10]
        wall_state = x[10:14]

        # Cost on feet position distance with respect to hips
        R = casadi.SX_eye(2)

        hip_pos_0 = x_CoM + casadi.mtimes(R, self.conf.hip_0)
        hip_pos_1 = x_CoM + casadi.mtimes(R, self.conf.hip_1)
        hip_pos_2 = x_CoM + casadi.mtimes(R, self.conf.hip_2)
        hip_pos_3 = x_CoM + casadi.mtimes(R, self.conf.hip_3)
        hip_pos_list = [hip_pos_0, hip_pos_1, hip_pos_2, hip_pos_3]

        hip_ps_mid_0 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_0)
        hip_ps_mid_1 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_1)
        hip_ps_mid_2 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_2)
        hip_ps_mid_3 = x_CoM + casadi.mtimes(R, self.conf.hip_mid_3)
        hip_ps_mid_list = [hip_ps_mid_0, hip_ps_mid_1, hip_ps_mid_2, hip_ps_mid_3]
        check_pos = hip_pos_list + hip_ps_mid_list
        
        ellipses = ((p_ee[0] - obstacle_state[0]) ** 2) / ((self.conf.A5 / 2) ** 2) + ((p_ee[1] - obstacle_state[1]) ** 2) / ((self.conf.B5 / 2) ** 2)

        for i in range(4):
            ellipses = casadi.vertcat(ellipses,
            ((check_pos[i][0]-obstacle_state[0])**2)/((self.conf.A5/2)**2) + ((check_pos[i][1]-obstacle_state[1])**2)/((self.conf.B5/2)**2),
            ((check_pos[i][0]-                  wall_state[0])**2)/((                 self.conf.A1/2)**2) + ((check_pos[i][1]-(wall_state[1]+wall_state[3])/2)**2)/(((wall_state[1]-wall_state[3])/2)**2),
            ((check_pos[i][0]-(wall_state[0]+wall_state[2])/2)**2)/(((wall_state[0]-wall_state[2])/2)**2) + ((check_pos[i][1]-                  wall_state[1])**2)/((                 self.conf.B2/2)**2),
            ((check_pos[i][0]-                  wall_state[2])**2)/((                 self.conf.A3/2)**2) + ((check_pos[i][1]-(wall_state[1]+wall_state[3])/2)**2)/(((wall_state[1]-wall_state[3])/2)**2),
            ((check_pos[i][0]-(wall_state[0]+wall_state[2])/2)**2)/(((wall_state[0]-wall_state[2])/2)**2) + ((check_pos[i][1]-                  wall_state[3])**2)/((                 self.conf.B4/2)**2)
            )  
        feasible_flags = casadi.mmin(ellipses) > 1

        return feasible_flags   