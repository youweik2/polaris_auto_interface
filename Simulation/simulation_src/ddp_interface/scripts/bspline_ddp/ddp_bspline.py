import numpy as np
from time import time
import logging
import numpy as np
from numpy.linalg import inv, solve
from casadi import MX, Function, jacobian, horzcat, DM
import casadi as ca

from collections.abc import Callable
from numpy.typing import ArrayLike

class DDPOptimizer:
    """Finite horizon Discrete-time Differential Dynamic Programming (DDP)"""

    def __init__(
        self,
        Nx: int,
        Nu: int,
        dynamics: Callable,
        inst_cost: Callable,
        terminal_cost: Callable,
        tolerance: float = 1e-5,
        max_iters: int = 500,
        with_hessians: bool = True,
        constrain: bool = True,
        alphas: ArrayLike = [1.0],
    ):
        """
        Instantiates a DDP Optimizer and pre-computes the dynamics
        and cost derivatives without doing any optimization/solving.
        """
        assert tolerance > 0
        assert max_iters > 0

        self.Nx = Nx
        self.Nu = Nu
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.with_hessians = with_hessians
        self.constrain = constrain
        self.alphas = alphas

        # Pre-compute derivatives for dynamics and costs using CasADi
        x = MX.sym("x", Nx)  # State vector
        u = MX.sym("u", Nu)  # Control vector
        x_goal = MX.sym("x_goal", Nx)  # Goal state

        # Dynamics
        dynamics_function = dynamics(x, u, constrain)
        fx = jacobian(dynamics_function, x)
        fu = jacobian(dynamics_function, u)

        self.f = Function("f", [x, u], [dynamics_function])
        self.fx = Function("fx", [x, u], [fx])
        self.fu = Function("fu", [x, u], [fu])

        if with_hessians:
            fxx = [jacobian(fx[:, i], x) for i in range(Nx)]
            fuu = [jacobian(fu[:, i], u) for i in range(Nu)]
            fux = [jacobian(fu[:, i], x) for i in range(Nu)]
            self.fxx = Function("fxx", [x, u], [horzcat(*fxx)])
            self.fuu = Function("fuu", [x, u], [horzcat(*fuu)])
            self.fux = Function("fux", [x, u], [horzcat(*fux)])

        # Instantaneous cost
        inst_cost_function = inst_cost(x, u, x_goal)
        gx = jacobian(inst_cost_function, x)
        gu = jacobian(inst_cost_function, u)
        gxx = [jacobian(gx[:, i], x) for i in range(Nx)]
        guu = [jacobian(gu[:, i], u) for i in range(Nu)]
        gux = [jacobian(gu[:, i], x) for i in range(Nu)]

        self.g = Function("g", [x, u, x_goal], [inst_cost_function])
        self.gx = Function("gx", [x, u, x_goal], [gx])
        self.gu = Function("gu", [x, u, x_goal], [gu])
        self.gxx = Function("gxx", [x, u, x_goal], [horzcat(*gxx)])
        self.guu = Function("guu", [x, u, x_goal], [horzcat(*guu)])
        self.gux = Function("gux", [x, u, x_goal], [horzcat(*gux)])

        # Terminal cost
        term_cost_function = terminal_cost(x, x_goal)
        hx = jacobian(term_cost_function, x)
        hxx = [jacobian(hx[:, i], x) for i in range(Nx)]

        self.h = Function("h", [x, x_goal], [term_cost_function])
        self.hx = Function("hx", [x, x_goal], [hx])
        self.hxx = Function("hxx", [x, x_goal], [horzcat(*hxx)])

    def optimize(
        self,
        x0: ArrayLike,
        x_goal: ArrayLike,
        N: int = None,
        U0: ArrayLike = None,
        full_output: bool = False,
        control_bounds: dict = None,
    ):
        """Optimize a trajectory given a starting state and a goal state."""
        start = time()
        x0 = np.array(x0)
        x_goal = np.array(x_goal)

        if U0 is not None:
            U = np.array(U0)
        else:
            assert N > 0
            U = np.random.uniform(-1.0, 1.0, (N, self.Nu))

        # Cost function
        def J(X, U):
            total_cost = 0.0
            for i in range(len(U)):
                total_cost += self.g(X[i], U[i], x_goal).toarray().item()
            total_cost += self.h(X[-1], x_goal).toarray().item()
            return float(total_cost)

        # Initialize trajectory
        X = np.zeros((N + 1, self.Nx))
        X[0] = x0.flatten()
        for i in range(len(U)):
            X[i + 1] = self.f(X[i], U[i]).toarray().flatten()

        last_cost = J(X, U)

        if full_output:
            X_hist = [X.copy()]
            U_hist = [U.copy()]
            cost_hist = [last_cost]

        # Main DDP loop
        for i in range(self.max_iters):
            # Backward pass
            Vx = self.hx(X[-1], x_goal).toarray().flatten()
            Vxx = self.hxx(X[-1], x_goal).toarray().reshape(self.Nx, self.Nx)

            Qus = np.zeros((N, self.Nu))
            Quus = np.zeros((N, self.Nu, self.Nu))
            Quxs = np.zeros((N, self.Nu, self.Nx))

            for t in reversed(range(N)):
                gx = self.gx(X[t], U[t], x_goal).toarray().flatten()
                gu = self.gu(X[t], U[t], x_goal).toarray().flatten()
                gxx = self.gxx(X[t], U[t], x_goal).toarray().reshape(self.Nx, self.Nx)
                gux = self.gux(X[t], U[t], x_goal).toarray().reshape(self.Nu, self.Nx)
                guu = self.guu(X[t], U[t], x_goal).toarray().reshape(self.Nu, self.Nu)
                fx = self.fx(X[t], U[t]).toarray()
                fu = self.fu(X[t], U[t]).toarray()

                Qx = gx + fx.T @ Vx
                Qu = gu + fu.T @ Vx
                Qxx = gxx + fx.T @ Vxx @ fx
                Quu = guu + fu.T @ Vxx @ fu
                Qux = gux + fu.T @ Vxx @ fx

                Qus[t] = Qu
                Quus[t] = Quu
                Quxs[t] = Qux

                Quu_inv = np.linalg.inv(Quu)
                Vx = Qx - Qux.T @ Quu_inv @ Qu
                Vxx = Qxx - Qux.T @ Quu_inv @ Qux

            # Forward pass with backtracking and control constraints
            for k, alpha in enumerate(self.alphas):
                X_star = np.zeros_like(X)
                U_star = np.zeros_like(U)
                X_star[0] = X[0].copy()

                for t in range(N):
                    error = X_star[t] - X[t]
                    U_star[t] = U[t] - np.linalg.solve(Quus[t], alpha * Qus[t] + Quxs[t] @ error)
                    
                    # Clip control inputs to enforce constraints
                    if control_bounds is not None:
                        U_star[t, 0] = np.clip(U_star[t, 0], control_bounds['a_min'], control_bounds['a_max'])
                        U_star[t, 1] = np.clip(U_star[t, 1], control_bounds['fai_min'], control_bounds['fai_max'])
                    
                    X_star[t + 1] = self.f(X_star[t], U_star[t]).toarray().flatten()

                # Update cost metric to see if we're doing well
                total_cost = J(X_star, U_star)
                if total_cost < last_cost:
                    logging.info(
                        f"Accepting new solution with J={total_cost} alpha={alpha:.2f} and {k} backtracks"
                    )
                    X = X_star
                    U = U_star
                    last_cost = total_cost
                    break

            if full_output:
                X_hist.append(X.copy())
                U_hist.append(U.copy())
                cost_hist.append(last_cost)

            if abs(last_cost - J(X, U)) < self.tolerance:
                break

        if full_output:
            return X, U, X_hist, U_hist, cost_hist

        return X, U

class BsplineDDP(DDPOptimizer):
    def __init__(self, Nx, Nu, dynamics, inst_cost, terminal_cost, 
                bspline_config, **kwargs):
        super().__init__(Nx, Nu, dynamics, inst_cost, terminal_cost, **kwargs)
        self.bs = bspline_config
        
    def optimize(self, x0, x_goal, q_init, **kwargs):
        # Generate initial controls from B-spline
        U0 = self.bs.points_to_controls(q_init)
        # print(U0)
        # Get correct number of control steps from B-spline
        N = len(U0)  # This fixes the index mismatch
        
        # Run original DDP optimization with proper N
        result = super().optimize(x0, x_goal, N=N, U0=U0, **kwargs)
        
        # Unpack results based on full_output flag
        if len(result) == 5:  # full_output=True
            X, U, X_hist, U_hist, cost_hist = result
            q_opt = self.bs.controls_to_points(U)
            return X, U, q_opt, X_hist, U_hist, cost_hist
        else:  # normal output
            X, U = result
            q_opt = self.bs.controls_to_points(U)
            return X, U, q_opt
            
    def project_controls(self, U):
        """Project arbitrary controls to B-spline space"""
        return self.bs.controls_to_points(U)