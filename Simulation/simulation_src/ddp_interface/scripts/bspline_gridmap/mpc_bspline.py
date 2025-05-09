import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from bspline import BSplineGenerator
from ddp_bspline import BsplineDDP
import tqdm

# Grid Map configuration
grid_matrix = [0, 100, 100, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1,\
            -1, 0, 100, -1, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
            0, 100, 0, 0, 0, 100, 100, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 100, 0, -1, -1, 0, 0, 0, 0, \
            0, 0, 100, -1, 100, 100, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 100, \
            100, 100, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 100, 100, 100, 0, 0,\
                0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
            0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
            0, 0, -1, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, \
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, \
            100, 0, -1, 0, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 100, 100, 100, \
            -1, -1, 0, 100, 100, 0, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
            -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

# grid_matrix = [0] * 440

grid_matrix = np.array(grid_matrix)

grid_input = grid_matrix.reshape((20, 22))

def get_gvalue(cur_x, cur_y, cx, cy, grid):

    hpg, wpg  = 22, 20
    h = 0.5

    # Reshape the flattened grid into a 50x50 symbolic matrix
    gridmap = ca.reshape(grid, wpg, hpg)

    # Compute symbolic grid indices
    grid_x = ca.fmax(ca.floor((cur_x + cx) / h), 0)
    grid_y = ca.fmax(ca.floor((cur_y + cy) / h), 0)

    def mappingm(matrix, row_idx, col_idx):
        result = 0
        for i in range(wpg):
            for j in range(hpg):
                result += ca.if_else(
                    ca.logic_and(row_idx == i, col_idx == j),
                    ca.if_else(matrix[i, j] == -1, 30, matrix[i, j]),
                    0
                )
        return result

    # Access grid values using the updated symbolic_lookup
    gxy = ca.if_else(ca.logic_or(grid_x >= hpg - 1, grid_y >= wpg - 1), 30, mappingm(gridmap, grid_y, grid_x))
    gxpy = ca.if_else(ca.logic_or(grid_x >= hpg - 2, grid_y >= wpg - 1), 30, mappingm(gridmap, grid_y, grid_x + 1))
    gxyp = ca.if_else(ca.logic_or(grid_x >= hpg - 1, grid_y >= wpg - 2), 30, mappingm(gridmap, grid_y + 1, grid_x))
    gxpyp = ca.if_else(ca.logic_or(grid_x >= hpg - 2, grid_y >= wpg - 2), 30, mappingm(gridmap, grid_y + 1, grid_x + 1))

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


def f(x, u, constrain=True):
    dt = 0.05
    theta = x[2]
    return ca.vertcat(
        x[0] + u[0] * ca.cos(theta) * dt,
        x[1] + u[0] * ca.sin(theta) * dt,
        x[2] + u[1] * dt
    )

def Phi(x, x_goal):
    Qf = np.diag([100, 100, 10])
    return (x - x_goal).T @ Qf @ (x - x_goal)

# def L(x, u, x_goal, old_omega=0.0):
    # State error cost
    error = x - x_goal
    Q = np.diag([100, 10, 0.1])
    state_cost = error.T @ Q @ error
    
    # Control cost
    R = np.diag([1, 1])
    control_cost = u.T @ R @ u
    
    # Obstacle cost
    obstacle_cost = 0
    for ox, oy, r in obstacles:
        dist_sq = (x[0]-ox)**2 + (x[1]-oy)**2
        safe_dist = (r + 0.12)**2
        obstacle_cost += 1e2 * ca.if_else(dist_sq < safe_dist, 1/(dist_sq + 1e-4), 0)
    
    # Control constraints
    v = u[0]
    omega = u[1]
    barrier_v = ca.fmax(0, (v-3.0)**2) + ca.fmax(0, (-3.0-v)**2)
    barrier_omega = ca.fmax(0, (omega-0.5)**2) + ca.fmax(0, (-0.5-omega)**2)
    
    return state_cost + control_cost + 10*obstacle_cost + 1e-3*(barrier_v + barrier_omega)

# def L(x, u, x_goal, old_omega=0.0):
    # State error cost (normalized)
    error = x - x_goal
    Q = ca.diag(ca.DM([10.0, 10.0, 0.1]))  # Reduced position weights
    state_cost = error.T @ Q @ error
    
    # Control cost (smoother penalties)
    R = ca.diag(ca.DM([1, 10.0]))  # Reduced control effort penalty
    control_cost = u.T @ R @ u
    
    # Progressive obstacle penalty
    obstacle_cost = 0
    # for ox, oy, r in obstacles:
    #     dist = ca.sqrt((x[0]-ox)**2 + (x[1]-oy)**2)
    #     safe_dist = r + 0.15  # Increased safety margin
        
    #     # Sigmoid-based penalty with gradual activation
    #     penalty_scale = 1e4
    #     slope = 2.0  # Controls transition sharpness
    #     obstacle_cost += penalty_scale * ca.log(1 + ca.exp(-slope*(dist - safe_dist)))
    

    for ox, oy, r in obstacles:
        dist = ca.sqrt((x[0]-ox)**2 + (x[1]-oy)**2)
        safe_dist = r + 0.5  # Increased safety margin
        
        # Smooth penalty with exponential activation
        penalty_scale = 100.0 
        steepness = 5.0
        obstacle_cost += penalty_scale * ca.log(1 + ca.exp(-steepness*(dist - safe_dist)))


    # Progressive control constraints
    v = u[0]
    omega = u[1]
    barrier_v = 1e-2*(ca.fmax(0, v-2.5)**3 + ca.fmax(0, -2.5-v)**3)  # Cubic penalty
    barrier_omega = 1e-2*(ca.fmax(0, omega-0.4)**3 + ca.fmax(0, -0.4-omega)**3)
    
    # Directional encouragement
    goal_dir = x_goal[:2] - x[:2]
    movement_dir = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    alignment = 0.1*(1 - goal_dir.T @ movement_dir/(ca.norm_2(goal_dir)+1e-6))
    
    return state_cost + control_cost + obstacle_cost + 10*barrier_v + 10*barrier_omega + alignment

def L(x, u, x_goal, old_omega=0.0):
    # State error cost (reduce position dominance)
    error = x - x_goal
    Q = ca.diag(ca.DM([10.0, 10.0, 0.1]))  # Keep orientation weight low
    state_cost = error.T @ Q @ error
    
    # Control cost (keep controls smooth)
    R = ca.diag(ca.DM([0.1, 0.1]))  # Reduced control penalties
    control_cost = u.T @ R @ u
    
    # Obstacle cost (revised for effective avoidance)
    obstacle_cost = 0

    gvalue = get_gvalue(x[1], x[0], -1.2825, 5, grid_input)

    # # method 1 - direct
    penalty_scale = 100.0
    obstacle_cost +=  penalty_scale * gvalue  
    
    # method 2 - Gradient-aware penalty function
    # penalty_scale = 1000.0  # Increased penalty magnitude
    # steepness = 10.0  # Sharper transition
    # obstacle_cost += penalty_scale * ca.exp(-steepness * (40 - gvalue))
    
    # Directional guidance (enhanced)
    goal_dir = x_goal[:2] - x[:2]
    movement_dir = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    alignment = 0.5 * (1 - goal_dir.T @ movement_dir/(ca.norm_2(goal_dir)+1e-6))
    
    # Progressive control constraints
    v = u[0]
    omega = u[1]
    barrier_v = 1e-4*(ca.fmax(0, v-2.0)**3 + ca.fmax(0, -2.0-v)**3)
    barrier_omega = 1e-4*(ca.fmax(0, omega-0.4)**3 + ca.fmax(0, -0.4-omega)**3)

    y = x[1]
    y_min = -5.0  # Lower y-boundary
    y_max = 5.0   # Upper y-boundary
    boundary_safety_margin = 0.3  # Soft margin for constraint
    
    # Smooth boundary penalty using logistic functions
    boundary_scale = 200.0  # Strength of boundary enforcement
    boundary_steepness = 10.0  # How quickly penalty increases at boundaries
    
    # Lower boundary penalty
    lower_penalty = ca.log(1 + ca.exp(boundary_steepness * (y_min + boundary_safety_margin - y)))
    
    # Upper boundary penalty
    upper_penalty = ca.log(1 + ca.exp(boundary_steepness * (y - (y_max - boundary_safety_margin))))
    
    boundary_cost = boundary_scale * (lower_penalty + upper_penalty)
    
    return state_cost + control_cost + obstacle_cost + alignment + barrier_v + barrier_omega + boundary_cost

def main():
    # B-spline configuration: 8 control points -> 60 controls
    bs = BSplineGenerator(
        degree=3,
        num_ctrl_points=8,
        time_horizon=3.0,
        control_dim=2,
        num_samples=60
    )
    
    # Initialize DDP solver
    ddp = BsplineDDP(
        Nx=3, Nu=2,
        dynamics=f,
        inst_cost=lambda x, u, x_goal: L(x, u, x_goal),
        terminal_cost=Phi,
        bspline_config=bs
    )
    
    
    # MPC parameters
    x_current = ca.DM([0.9, 1, 0])
    x_goal = ca.DM([50, -1, 0])
    q = np.ones((bs.num_ctrl_points, 2))
    
    X_hist = [x_current.full().flatten()]
    U_hist = []

    u_old = np.array([1.0, 0.0])


    for step in tqdm.tqdm(range(1000)):
        # Run B-spline constrained DDP
        X_opt, U_opt, q = ddp.optimize(
            x_current, x_goal, q,
            control_bounds={'v_min': -3.0, 'v_max': 3.0,
                           'omega_min': -0.5, 'omega_max': 0.5},
            full_output=False
        )
        
        # Apply first control
        u_apply = U_opt[0]

        # Clip control inputs
        u_apply[0] = np.clip(u_apply[0], -3.0, 3.0)
        u_apply[1] = np.clip(u_apply[1], -0.5, 0.5)

        # Clip control based on the continuous change
        u_apply[1] = np.clip(u_apply[1], u_old[1]-0.15, u_old[1]+0.15)
        u_apply[0] = np.clip(u_apply[0], u_old[0]-0.5, u_old[0]+0.5)
        u_old = u_apply
        x_current = f(x_current, ca.DM(u_apply))
        X_hist.append(x_current.full().flatten())
        U_hist.append(u_apply)

        # Shift control points
        q = np.roll(q, -1, axis=0)
        q[-1] = q[-2]  # Maintain continuity
        
        # Termination check
        if ca.norm_2(x_current[:2] - x_goal[:2]) < 1.0:
            print("Goal reached!")
            break
    
    # Visualization
    X_hist = np.array(X_hist)
    U_hist = np.array(U_hist)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    # Trajectory plot
    ax[0].plot(X_hist[:,0], X_hist[:,1], 'b-', label='Trajectory')
    ax[0].plot(x_goal[0], x_goal[1], 'g*', markersize=15, label='Goal')

    # Define the square size in meters
    square_size = 0.5

    # Create a color map where 0 is white, 50 is gray, and 100 is black
    color_map = {0: "white", -1: "gray", 100: "black"}
    cx, cy  = 1.2825, -5

    for i in range(len(grid_input)):
        for j in range(len(grid_input[i])):
            color = color_map[grid_input[i][j]]
            # Draw each square with the appropriate color
            ax[0].add_patch(plt.Rectangle((cx + j * square_size, cy + i * square_size), square_size, square_size, color=color))

    ax[0].legend()
    ax[0].set_aspect('equal')
    
    # Control inputs
    ax[1].plot(U_hist[:,0], label='Velocity')
    ax[1].plot(U_hist[:,1], label='Angular Velocity')
    ax[1].legend()
    
    # Cost plot
    ax[2].plot(np.linalg.norm(X_hist - x_goal.full().T, axis=1))
    ax[2].set_ylabel('Distance to Goal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()