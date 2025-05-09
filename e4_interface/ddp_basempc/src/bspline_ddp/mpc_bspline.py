import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from bspline import BSplineGenerator
from ddp_bspline import BsplineDDP
import tqdm

# Obstacle configuration
obstacles = [
   (0.0, 15, 1.25), (1.0, 20, 1) , (-1, 25, 1)  #(0.0, 15, 1),  (1, 20, 1), (-1, 25, 1) # , (-2.0, 25, 1.2), (1.0, 28, 1.0)
]

def wrap_to_pi(theta):
    wrapped_theta = theta - 2 * ca.pi * ca.floor((theta + ca.pi) / (2 * ca.pi))
    return wrapped_theta

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
    Qf = np.diag([100, 100, 30.0, 0.0, 0.1])
    return (x - x_goal).T @ Qf @ (x - x_goal)

def L(x, u, x_goal):
    error = x - x_goal
    Q = ca.diag(ca.DM([30.0, 30.0, 0.0, 0.0, 1.0]))  
    state_cost = error.T @ Q @ error
    
    R = ca.diag(ca.DM([0.1, 0.1]))  
    control_cost = u.T @ R @ u
    
    obstacle_cost = 0
    for ox, oy, r in obstacles:
        dist = ca.sqrt((x[0] - ox)**2 + (x[1] - oy)**2)
        safe_dist = r + 1.0  
        penalty_scale = 800.0  
        steepness = 10.0  
        obstacle_cost += penalty_scale * ca.exp(-steepness * (dist - safe_dist))
    
    goal_dir = x_goal[:2] - x[:2]
    movement_dir = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    alignment = 0.5 * (1 - goal_dir.T @ movement_dir / (ca.norm_2(goal_dir) + 1e-6))
    
    omega = u[1]
    barrier_omega = 0.0
    px = x[0]
    x_min, x_max = -5.0, 5.0  
    boundary_safety_margin = 0.3  

    boundary_scale = 200.0  
    boundary_steepness = 10.0  

    lower_penalty = ca.log(1 + ca.exp(boundary_steepness * (x_min + boundary_safety_margin - px)))
    upper_penalty = ca.log(1 + ca.exp(boundary_steepness * (px - (x_max - boundary_safety_margin))))
    
    boundary_cost = boundary_scale * (lower_penalty + upper_penalty)

    phi = x[3]
    phi_min, phi_max = -np.pi/6, np.pi/6
    lower_phi_penalty = ca.log(1 + ca.exp(boundary_steepness * (phi_min + boundary_safety_margin - phi)))
    upper_phi_penalty = ca.log(1 + ca.exp(boundary_steepness * (phi - (phi_max - boundary_safety_margin))))
    phi_boundary_cost = boundary_scale * (lower_phi_penalty + upper_phi_penalty)

    v = x[4]
    v_min, v_max = -0.0, 4.0
    lower_v_penalty = ca.log(1 + ca.exp(boundary_steepness * (v_min + boundary_safety_margin - v)))
    upper_v_penalty = ca.log(1 + ca.exp(boundary_steepness * (v - (v_max - boundary_safety_margin))))
    v_boundary_cost = boundary_scale * (lower_v_penalty + upper_v_penalty)

    return 15 * state_cost + 0.6 * control_cost + 0.98 * obstacle_cost + 1.03 * alignment + 1.05 * boundary_cost + 11.25 * phi_boundary_cost + 0.7 * v_boundary_cost
# 15 * state_cost + 0.6 * control_cost + 0.98 * obstacle_cost + 1.03 * alignment + 1.05 * boundary_cost + 11.25 * phi_boundary_cost + 0.72 * v_boundary_cost

def L_calculation(x, u, x_goal):
    """
    计算所有 cost 组件并返回
    """
    error = x - x_goal
    Q = ca.diag(ca.DM([30.0, 30.0, 0.0, 0.0, 1.0]))  
    state_cost = error.T @ Q @ error

    R = ca.diag(ca.DM([0.1, 0.1]))  
    control_cost = u.T @ R @ u

    # 计算障碍物成本
    obstacle_cost = 0
    for ox, oy, r in obstacles:
        dist = ca.sqrt((x[0] - ox)**2 + (x[1] - oy)**2)
        safe_dist = r + 1.0  
        penalty_scale = 800.0  
        steepness = 10.0  
        obstacle_cost += penalty_scale * ca.exp(-steepness * (dist - safe_dist))

    # 计算对齐成本（朝向目标方向）
    goal_dir = x_goal[:2] - x[:2]
    movement_dir = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    alignment = 0.5 * (1 - goal_dir.T @ movement_dir / (ca.norm_2(goal_dir) + 1e-6))

    omega = u[1]
    barrier_omega = 0.0
    px = x[0]
    x_min, x_max = -5.0, 5.0  
    boundary_safety_margin = 0.3  

    boundary_scale = 200.0  
    boundary_steepness = 10.0  

    # 计算边界成本
    lower_penalty = ca.log(1 + ca.exp(boundary_steepness * (x_min + boundary_safety_margin - px)))
    upper_penalty = ca.log(1 + ca.exp(boundary_steepness * (px - (x_max - boundary_safety_margin))))
    boundary_cost = boundary_scale * (lower_penalty + upper_penalty)

    # 计算 phi 方向角的边界成本
    phi = x[3]
    phi_min, phi_max = -np.pi/6, np.pi/6
    lower_phi_penalty = ca.log(1 + ca.exp(boundary_steepness * (phi_min + boundary_safety_margin - phi)))
    upper_phi_penalty = ca.log(1 + ca.exp(boundary_steepness * (phi - (phi_max - boundary_safety_margin))))
    phi_boundary_cost = boundary_scale * (lower_phi_penalty + upper_phi_penalty)

    # 计算速度边界成本
    v = x[4]
    v_min, v_max = -0.0, 6.0
    lower_v_penalty = ca.log(1 + ca.exp(boundary_steepness * (v_min + boundary_safety_margin - v)))
    upper_v_penalty = ca.log(1 + ca.exp(boundary_steepness * (v - (v_max - boundary_safety_margin))))
    v_boundary_cost = boundary_scale * (lower_v_penalty + upper_v_penalty)

    # 计算总成本
    total_cost = (
        10 * state_cost + 
        10 * control_cost + 
        obstacle_cost + 
        alignment + 
        boundary_cost + 
        phi_boundary_cost + 
        0.01 * v_boundary_cost
    )

    # ✅ 转换 CasADi.DM 为 float 以确保 Python 兼容性
    total_cost = float(total_cost.full()[0, 0])
    state_cost = float(state_cost.full()[0, 0])
    control_cost = float(control_cost.full()[0, 0])
    obstacle_cost = float(obstacle_cost.full()[0, 0])
    alignment = float(alignment.full()[0, 0])
    boundary_cost = float(boundary_cost.full()[0, 0])
    phi_boundary_cost = float(phi_boundary_cost.full()[0, 0])
    v_boundary_cost = float(v_boundary_cost.full()[0, 0])

    # ✅ 返回所有成本组件
    return total_cost, state_cost, control_cost, obstacle_cost, alignment, boundary_cost, phi_boundary_cost, v_boundary_cost


def main():
    bs = BSplineGenerator(
        degree=5,
        num_ctrl_points=8,
        time_horizon=4.0,
        control_dim=2,
        num_samples=50
    )
    
    ddp = BsplineDDP(
        Nx=5, Nu=2,
        dynamics=f,
        inst_cost=lambda x, u, x_goal: L(x, u, x_goal),
        terminal_cost=Phi,
        bspline_config=bs
    )
    
    x_current = ca.DM([-0.0, -1.0, np.pi/2, 0, 0])
    x_goal = ca.DM([-0.0, 40, np.pi/2, 0, 0])
    q = np.zeros((bs.num_ctrl_points, 2))

    X_hist = [x_current.full().flatten()]
    U_hist = []

    total_cost_hist = []
    state_cost_hist = []
    control_cost_hist = []
    obstacle_cost_hist = []
    alignment_hist = []
    boundary_cost_hist = []
    phi_boundary_cost_hist = []
    v_boundary_cost_hist = []


    for step in tqdm.tqdm(range(1000)):
        X_opt, U_opt, q = ddp.optimize(
            x_current, x_goal, q,
            control_bounds={'a_min': -2.0, 'a_max': 0.8, 'fai_min': -np.pi/6, 'fai_max': np.pi/6},
            full_output=False
        )

        u_apply = U_opt[0]
        u_apply = np.clip(u_apply, [-2.0, -np.pi/6], [0.8, np.pi/6])
        x_current[2] = wrap_to_pi(x_current[2])

        x_current = f(x_current, ca.DM(u_apply))

        # Shift control points
        q = np.roll(q, -1, axis=0)
        q[-1] = q[-2]  # Maintain continuity


        # 计算所有成本
        total_cost, state_cost, control_cost, obstacle_cost, alignment, boundary_cost, phi_boundary_cost, v_boundary_cost = \
            L_calculation(x_current, ca.DM(u_apply), x_goal)

        # 记录历史数据
        total_cost_hist.append(total_cost)
        state_cost_hist.append(state_cost)
        control_cost_hist.append(control_cost)
        obstacle_cost_hist.append(obstacle_cost)
        alignment_hist.append(alignment)
        boundary_cost_hist.append(boundary_cost)
        phi_boundary_cost_hist.append(phi_boundary_cost)
        v_boundary_cost_hist.append(v_boundary_cost)


        X_hist.append(x_current.full().flatten())
        U_hist.append(u_apply)

        if x_current[1] > 40:
            print("Goal reached!")
            break

    X_hist = np.array(X_hist)
    U_hist = np.array(U_hist)

    fig, ax = plt.subplots(4, 1, figsize=(10, 16)) 

    ax[0].plot(X_hist[:,1], X_hist[:,0], 'b-', label='Trajectory')
    ax[0].plot(x_goal[1], x_goal[0], 'g*', markersize=15, label='Goal')
    for ox, oy, r in obstacles:
        ax[0].add_patch(plt.Circle((oy, ox), r, color='r', alpha=0.3))
    ax[0].legend()
    ax[0].set_aspect('equal')

    ax[1].plot(U_hist[:,0], label='Acceleration')
    ax[1].plot(U_hist[:,1], label='Omega')
    ax[1].plot(X_hist[:,2], label='Theta')
    ax[1].plot(X_hist[:,3], label='Phi')
    ax[1].plot(X_hist[:,4], label='Velocity')
    ax[1].legend()

    ax[2].plot(np.linalg.norm(X_hist - x_goal.full().T, axis=1))
    ax[2].set_ylabel('Distance to Goal')

    # ax[3].plot(total_cost_hist, label="Total Cost", linewidth=2, color='black')
    ax[3].plot(state_cost_hist, label="State Cost", linestyle="--", color='blue')
    ax[3].plot(control_cost_hist, label="Control Cost", linestyle=":", color='green')
    ax[3].plot(obstacle_cost_hist, label="Obstacle Cost", linestyle="-.", color='red')
    ax[3].plot(alignment_hist, label="Alignment Cost", linestyle="--", color='purple')
    ax[3].plot(boundary_cost_hist, label="Boundary Cost", linestyle=":", color='orange')
    ax[3].plot(phi_boundary_cost_hist, label="Phi Boundary Cost", linestyle="-.", color='brown')
    ax[3].plot(v_boundary_cost_hist, label="Velocity Boundary Cost", linestyle="--", color='cyan')
    ax[3].legend()


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
