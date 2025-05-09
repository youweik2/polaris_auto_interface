#!/usr/bin/env python
# Acados adds
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as ca

# Common lib
import numpy as np
import scipy.linalg
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import math

class GemCarModel(object):
    def __init__(self,):

        model = AcadosModel()
        constraint = ca.types.SimpleNamespace()
        length = 2.565

        # control inputs
        a = ca.SX.sym('accel')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(a, omega) # fai: front angle

        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')
        fai = ca.SX.sym('fai')
        states = ca.vertcat(x, y, theta, fai, v)

        # dynamic
        rhs = [v*ca.cos(theta), v*ca.sin(theta), v*ca.tan(fai)/length, omega, a] # v*ca.tan(fai)/length -> v*fai/length as |fai| < 30 degree

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        # other settings
        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'GemCarModel'

        # constraints
        constraint.a_max = 0.8
        constraint.a_min = -1.5
        constraint.omega_max = np.pi / 6   # np.pi/5.2
        constraint.omega_min = -np.pi/ 6 # -np.pi
        constraint.expr = ca.vcat([a, omega])

        self.model = model
        self.constraint = constraint

class GemCarOptimizer(object):

    def __init__(self, m_model, m_constraint, t_horizon, dt, grid, target):

        model = m_model

        # Grid dimensions
        self.wpg = 55
        self.hpg = 78
        self.rtg = grid  # Real-time grid

        self.T = t_horizon
        self.dt = dt
        self.N = int(t_horizon / dt)

        # Car Info
        self.car_width = 1.5
        self.car_length = 2.565
        self.Epi = 3000

        self.target_x = target[0]
        self.target_y = target[1]
        self.target_theta = target[2]
        self.target_fai = target[3]
        self.target_velocity = target[4]

        # Obstacle set here
        self.circle_obstacles_1 = {'x': 0, 'y': 15, 'r': 1}
        self.circle_obstacles_2 = {'x': 1, 'y': 20, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 25, 'r': 1.0}

        self.plot_figures = True

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = './acados_models'
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # Acados degrees (necessary)
        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = len(model.p) # number of extra parameters (0 here)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        grid_1d = self.rtg[0: self.wpg * self.hpg]
        self.grid = grid_1d
        self.gridplot = grid_1d.reshape(55, 78)
        self.cx = self.rtg[self.wpg * self.hpg]
        self.cy = self.rtg[self.wpg * self.hpg + 1]

        # Cost settings ***
        Q = np.diag([10.0, 10.0, 5.0, 25.0, 5.0])  # States
        R = np.array([[25.0, 0.0],[0.0, 80.0]])            # Controls

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = np.diag([10.0, 10.0, 0.8, 0.8, 4.0])
        ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # set constraints
        ocp.constraints.lbu = np.array([m_constraint.a_min, m_constraint.omega_min])
        ocp.constraints.ubu = np.array([m_constraint.a_max, m_constraint.omega_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbx = np.array([-10, -100, -np.pi, -np.pi/5.2, 0])
        ocp.constraints.ubx = np.array([10, 100, np.pi, np.pi/5.2, 3])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])

        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)

        # obstacles
        x_ = ocp.model.x[0]
        y_ = ocp.model.x[1]


        con_h_expr = []

        safe_condition = [(0,0)]# [(1, 0), (-1, 0), (0, 1), (0, 0)] #[(0,0)]

        for xp, yp in safe_condition:
            grid_value = 73 - self.get_gvalue(x_ + xp, y_ + yp, -self.cx + 13.75, -self.cy, self.grid)
            con_h_expr.append(grid_value)

        if con_h_expr:
            ocp.model.con_h_expr = ca.vertcat(*con_h_expr)
            ocp.constraints.lh = np.zeros((len(con_h_expr),))
            ocp.constraints.uh = 10 * np.ones((len(con_h_expr),))

            # slack variable configuration
            nsh = len(con_h_expr)
            ocp.constraints.lsh = np.zeros(nsh)
            ocp.constraints.ush = np.zeros(nsh)
            ocp.constraints.idxsh = np.array(range(nsh))

            ns = len(con_h_expr)
            ocp.cost.zl = 10 * np.ones((ns,))
            ocp.cost.Zl = 10 * np.ones((ns,))
            ocp.cost.zu = 0 * np.ones((ns,))
            ocp.cost.Zu = 0 * np.ones((ns,))


        # initial state
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'EXACT'
        ocp.solver_options.integrator_type = 'ERK' # 'IRK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' #'SQP_RTI'

        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def get_gvalue(self, cur_x, cur_y, cx, cy, grid):

        hpg, wpg  = 78, 55
        h = 0.5

        # Reshape the flattened grid into a 50x50 symbolic matrix
        gridmap = ca.reshape(grid, hpg, wpg)

        # Compute symbolic grid indices
        grid_x = ca.fmax(ca.floor((cur_x + cx) / h), 0)
        grid_y = ca.fmax(ca.floor((cur_y + cy) / h), 0)

        def mappingm(matrix, row_idx, col_idx):
            result = 0
            for i in range(hpg):
                for j in range(wpg):
                    result += ca.if_else(
                        ca.logic_and(row_idx == i, col_idx == j),
                        ca.if_else(matrix[i, j] == -1, 50, matrix[i, j]),
                        0
                    )
            return result
        
        # Access grid values using the updated symbolic_lookup
        gxy = ca.if_else(ca.logic_or(grid_x >= wpg - 1, grid_y >= hpg - 1), 50, mappingm(gridmap, grid_y, grid_x))
        gxpy = ca.if_else(ca.logic_or(grid_x >= wpg - 2, grid_y >= hpg - 1), 50, mappingm(gridmap, grid_y, grid_x + 1))
        gxyp = ca.if_else(ca.logic_or(grid_x >= wpg - 1, grid_y >= hpg - 2), 50, mappingm(gridmap, grid_y + 1, grid_x))
        gxpyp = ca.if_else(ca.logic_or(grid_x >= wpg - 2, grid_y >= hpg - 2), 50, mappingm(gridmap, grid_y + 1, grid_x + 1))

        # Compute weights
        I_x = ca.floor((cur_x + cx) / h)
        I_y = ca.floor((cur_y + cy) / h)
        R_x = (cur_x + cx) / h - I_x
        R_y = (cur_y + cy) / h - I_y

        # Symbolic matrix and vector operations
        m_x = ca.vertcat(1 - R_x, R_x)
        m_g = ca.horzcat(ca.vertcat(gxy, gxpy), ca.vertcat(gxyp, gxpyp))
        m_y = ca.vertcat(1 - R_y, R_y)

        # Compute the value
        g_value = ca.mtimes([m_x.T, m_g, m_y])
        return g_value


    def solve(self, x_real, y_real, theta_real, fai_real, velocity_real, a_real, o_real):

        # current state
        x0 = np.array([x_real, y_real, theta_real, fai_real, velocity_real])

        # terminal state
        xs = np.array([self.target_x, self.target_y, self.target_theta, self.target_fai, self.target_velocity])

        # setting
        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)  

        # reference
        vel_ref = np.zeros(int(self.N))

        for i in range(self.N):
            vel_ref[i] = velocity_real - velocity_real * i / self.N

        delta_x = self.target_x - x_real
        delta_y = self.target_y - y_real
        theta_between = np.abs(math.atan2(self.target_y - y_real, self.target_x - x_real))

        for i in range(self.N):
            if velocity_real > 2.5:
                xs_between = np.array([
                    delta_x / self.N * i + x_real,
                    delta_y / self.N * i + y_real,
                    theta_between,
                    fai_real,
                    0.5,
                    -1.0,
                    o_real
                ])
            elif velocity_real > 0.5:
                xs_between = np.array([
                    delta_x / self.N * i + x_real,
                    delta_y / self.N * i + y_real,
                    theta_between,
                    fai_real,
                    vel_ref[i],
                    a_real,
                    o_real
                ])
            else:
                xs_between = np.array([
                    delta_x / self.N * i + x_real,
                    delta_y / self.N * i + y_real,
                    theta_between,
                    fai_real,
                    2.0,
                    0.5,
                    0
                ])
            self.solver.set(i, 'yref', xs_between)
        self.solver.set(self.N, 'yref', xs)

        # Start Solving

        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)     
        status = self.solver.solve()

        if status != 0 :
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
        
        simX[0, :] = self.solver.get(0, 'x')

        for i in range(self.N):
            # solve ocp
            simU[i, :] = self.solver.get(i, 'u')
            simX[i+1, :] = self.solver.get(i+1, 'x')

        # next state
        next_x = simX[1, 0]
        next_y = simX[1, 1]
        next_theta = simX[1, 2]
        next_fai = simX[1, 3]
        next_vel = simX[1, 4]
        aim_a = simU[0, 0]
        aim_o = simU[0, 1]

        return next_x, next_y, next_theta, next_fai, next_vel, aim_a, aim_o


    # plot function
    def plot_results(self, start_x, start_y, theta_log, fai_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log):
        
        fig1, axs1 = plt.subplots(2, 1, figsize=(8, 6))

        a = np.arange(0, len(a_log)) * self.dt
        axs1[0].plot(a, a_log, 'r-', label='desired a')
        axs1[0].set_xlabel('time')
        axs1[0].set_ylabel('acceleration')
        axs1[0].legend()
        axs1[0].grid(True)

        axs1[1].plot(a, v_log, 'r-', label='current velocity')
        axs1[1].set_xlabel('time')
        axs1[1].set_ylabel('velocity')
        axs1[1].legend()
        axs1[1].grid(True)

        plt.tight_layout()
        plt.show()

        # 第二个图（theta 和 phi）
        t = np.arange(0, len(theta_log)) * self.dt
        fig2, axs2 = plt.subplots(2, 1, figsize=(8, 6))

        axs2[0].plot(t, theta_log, 'r-', label='desired theta')
        axs2[0].set_xlabel('time')
        axs2[0].set_ylabel('theta')
        axs2[0].legend()
        axs2[0].grid(True)

        axs2[1].plot(t, fai_log, 'r-', label='desired phi')
        axs2[1].set_xlabel('time')
        axs2[1].set_ylabel('phi')
        axs2[1].legend()
        axs2[1].grid(True)

        plt.tight_layout()
        plt.show()
        
        plt.plot(x_log, y_log, 'r-', label='desired path')
        plt.plot(x_real_log, y_real_log, color='b', linestyle='--', label='real path')
        plt.plot(self.target_x,self.target_y,'bo')
        plt.plot(start_x, start_y, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')

        # Define the square size in meters
        square_size = 0.5

        # Create a color map where 0 is white, 50 is gray, and 100 is black
        color_map = {0: "white", -1: "gray", 100: "black"}
        cx, cy  = -13.75, 0

        for i in range(len(self.gridplot)):
            for j in range(len(self.gridplot[i])):
                color = color_map[self.gridplot[i][j]]
                # Draw each square with the appropriate color
                plt.gca().add_patch(plt.Rectangle((cx + i * square_size, cy + j * square_size), square_size, square_size, color=color))

        plt.axis('equal')
        plt.legend()
        plt.show()


    def main(self, x_init, y_init, theta_init, fai_init, velocity_init, a_init, omega_init):
                    
        x_0, y_0, theta, vel= x_init, y_init, theta_init, velocity_init
        x_real, y_real, theta_real, fai_real, vel_real, a_real, omega_real = x_init, y_init, theta_init, fai_init, velocity_init, a_init, omega_init

        x_log, y_log = [], []
        theta_log = []
        a_log = []

        x_real_log, y_real_log = [], []
        o_log, v_log = [], []
        fai_log = []

        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
            for i in tqdm(range(self.Epi)):

                try:
                    x_0, y_0, theta, fai, vel, a_0, o_0 = self.solve(x_real, y_real, theta_real, fai_real, vel_real, a_real, omega_real)
                    

                    x_real, y_real, theta_real, fai_real, vel_real, a_real = x_0, y_0, theta, fai, vel, a_0
                    omega_real =  o_0
                    
                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta_real)
                    fai_log.append(fai_real)
                    a_log.append(a_0)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    o_log.append(o_0)
                    v_log.append(vel)

                    if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 1:
                        # break
                        print("reach the target", theta)
                        if self.plot_figures == True:
                            self.plot_results(x_init, y_init, theta_log, fai_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log)
                        return [1, theta_log], x_log, y_log

                except RuntimeError:
                    print("Infesible", theta)
                    if self.plot_figures == True:
                        self.plot_results(x_init, y_init, theta_log, fai_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log)
                    return [0, theta_log], x_log, y_log

            print("not reach the target", theta)
            if self.plot_figures == True:
                self.plot_results(x_init, y_init, theta_log, fai_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log)
            return [0, theta_log], x_log, y_log

if __name__ == '__main__':

    data = np.loadtxt('./projected_map.csv', delimiter=',')

    flat_data = data.flatten()

    grid = flat_data.tolist()
    
    grid_full = grid
    grid_1d = np.array(grid)

    grid_full.append(0)
    grid_full.append(0)
    grid_full = np.array(grid_full)

    grid = (flat_data.reshape(55, 78))

    length = 2.565
    start_x, start_y, theta, vel, a0, o0 = 0.0, 0.0, np.pi/2, 0.001, 0, 0
    fai = 0.0
    terminal = np.array([0.0, 40.0, np.pi/2, 0.0, 0.0])


    car_model = GemCarModel()
    opt = GemCarOptimizer(m_model=car_model.model, 
                               m_constraint=car_model.constraint, grid = grid_full, t_horizon=2.0, dt=0.05, target=terminal)
    opt.main(start_x, start_y, theta, fai, vel, a0, o0)