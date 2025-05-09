import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from bspline import BSplineGenerator
from ddp_bspline import BsplineDDP
import tqdm

# Obstacle configuration
obstacles = [
    (0.2, 20, 1.2)# , (1.0, 30, 1.2) # , (18, 1, 1), (32, 0, 1.5)
]

def f(x, u, constrain=True):
    dt = 0.05
    length = 2.565
    return ca.vertcat(
        x[0] + x[4] * ca.cos(x[2]) * dt,
        x[1] + x[4] * ca.sin(x[2]) * dt,
        x[2] + x[4] * ca.tan(x[3]) / length * dt,
        x[3] + u[1] * dt,
        x[4] + u[0] * dt
    )

def Phi(x, x_goal):
    Qf = np.diag([120, 120, 35, 10, 15])
    return (x - x_goal).T @ Qf @ (x - x_goal)

def L(x, u, x_goal, old_omega=0.0):
    # State error cost (reduce position dominance)
    error = x - x_goal
    Q = ca.diag(ca.DM([30.0, 30.0, 0.05, 0.05, 0.01]))  # Keep orientation weight low
    state_cost = error.T @ Q @ error
    
    # Control cost (keep controls smooth)
    R = ca.diag(ca.DM([0.1, 0.1]))  # Reduced control penalties
    control_cost = u.T @ R @ u
    
    # Obstacle cost (revised for effective avoidance)
    obstacle_cost = 0
    for ox, oy, r in obstacles:
        dist = ca.sqrt((x[0]-ox)**2 + (x[1]-oy)**2)
        safe_dist = r + 0.8  # Increased safety margin
        
        # Gradient-aware penalty function
        penalty_scale = 800.0  # Increased penalty magnitude
        steepness = 10.0  # Sharper transition
        obstacle_cost += penalty_scale * ca.exp(-steepness * (dist - safe_dist))
    
    # Directional guidance (enhanced)
    goal_dir = x_goal[:2] - x[:2]
    movement_dir = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    alignment = 0.5 * (1 - goal_dir.T @ movement_dir/(ca.norm_2(goal_dir)+1e-6))
    
    # Progressive control constraints
    v = x[3]
    omega = u[1]
    barrier_v = 1e-4*(ca.fmax(0, v-2.0)**3 + ca.fmax(0, -2.0-v)**3)
    barrier_omega = 1e-4*(ca.fmax(0, omega-0.4)**3 + ca.fmax(0, -0.4-omega)**3)

    x = x[0]
    x_min = -5.0  # Lower y-boundary
    x_max = 5.0   # Upper y-boundary
    boundary_safety_margin = 0.3  # Soft margin for constraint
    
    # Smooth boundary penalty using logistic functions
    boundary_scale = 200.0  # Strength of boundary enforcement
    boundary_steepness = 10.0  # How quickly penalty increases at boundaries
    
    # Lower boundary penalty
    lower_penalty = ca.log(1 + ca.exp(boundary_steepness * (x_min + boundary_safety_margin - x)))
    
    # Upper boundary penalty
    upper_penalty = ca.log(1 + ca.exp(boundary_steepness * (x - (x_max - boundary_safety_margin))))
    
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
        Nx=5, Nu=2,
        dynamics=f,
        inst_cost=lambda x, u, x_goal: L(x, u, x_goal),
        terminal_cost=Phi,
        bspline_config=bs
    )
    
    # MPC parameters
    x_current = ca.DM([1, -2, np.pi/3, 0, 0])
    x_goal = ca.DM([-0.0, 40, np.pi/2, 0, 0])
    q = np.ones((bs.num_ctrl_points, 2))
    
    X_hist = [x_current.full().flatten()]
    U_hist = []

    u_old = np.array([1.0, 0.0])
    
    for step in tqdm.tqdm(range(1000)):
        # Run B-spline constrained DDP
        X_opt, U_opt, q = ddp.optimize(
            x_current, x_goal, q,
            control_bounds={'a_min': -2.0, 'a_max': 0.8,
                           'fai_min': -np.pi/6, 'fai_max': np.pi/6},
            full_output=False
        )
        
        # Apply first control
        u_apply = U_opt[0]

        # Clip control inputs
        u_apply[0] = np.clip(u_apply[0], -2.0, 0.8)
        u_apply[1] = np.clip(u_apply[1], -np.pi/6, np.pi/6)

        # Clip control based on the continuous change
        u_apply[1] = np.clip(u_apply[1], u_old[1]-0.3, u_old[1]+0.3)
        u_apply[0] = np.clip(u_apply[0], u_old[0]-1, u_old[0]+1)
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
    ax[0].plot(X_hist[:,1], X_hist[:,0], 'b-', label='Trajectory')
    ax[0].plot(x_goal[1], x_goal[0], 'g*', markersize=15, label='Goal')
    for ox, oy, r in obstacles:
        ax[0].add_patch(plt.Circle((oy, ox), r, color='r', alpha=0.3))
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