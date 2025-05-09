import numpy as np
import casadi as ca

class BSplineGenerator:
    def __init__(self, degree, num_ctrl_points, time_horizon, control_dim, num_samples):
        self.p = degree
        self.n = num_ctrl_points - 1
        self.T = time_horizon
        self.m = control_dim
        self.num_samples = num_samples
        self.num_ctrl_points = num_ctrl_points
        
        # Create clamped knot vector
        self.tau = np.concatenate([
            [0]*self.p,
            np.linspace(0, self.T, num_ctrl_points - degree + 1),
            [self.T]*self.p
        ])
        # # Scale knots to constraint space
        # self.tau = np.concatenate([
        #     [0]*self.p,
        #     np.linspace(v_min, v_max, self.num_segments + 1),
        #     [v_max]*self.p
        # ])
        
        # Cubic B-spline basis matrix
        self.M = np.array([
            [1, 3, 3, 1],
            [-3, -6, 3, 4],
            [3, 3, -6, -6],
            [-1, 0, 3, 1]
        ]) / 6.0  # Ensures 0 ≤ basis ≤1
        
        # Precompute basis matrix for control projection
        self.basis_matrix = self._create_basis_matrix()
        
    def _create_basis_matrix(self):
        """Create basis matrix for least-squares projection"""
        B = np.zeros((self.num_samples, self.num_ctrl_points))
        times = np.linspace(0, self.T, self.num_samples)
        
        for i, t in enumerate(times):
            seg_idx = int(np.clip(
                (t / self.T) * (self.num_ctrl_points - self.p),
                0, self.num_ctrl_points - self.p - 1e-9
            ))
            t_local = (t / self.T) * (self.num_ctrl_points - self.p) - seg_idx
            
            basis = np.array([
                (1 - t_local)**3 / 6,
                (3*t_local**3 - 6*t_local**2 + 4) / 6,
                (-3*t_local**3 + 3*t_local**2 + 3*t_local + 1) / 6,
                t_local**3 / 6
            ])
            
            B[i, seg_idx:seg_idx+4] = basis
            
        return B

    def symbolic_control(self, t, ctrl_points):
        """CasADi-compatible B-spline evaluation at time t"""
        t_norm = t / self.T
        seg_idx = ca.fmax(0, ca.fmin(
            self.n - self.p,
            ca.floor(t_norm * (self.num_ctrl_points - self.p))
        ))
        t_local = t_norm * (self.num_ctrl_points - self.p) - seg_idx
        
        basis = ca.vertcat(1, t_local, t_local**2, t_local**3)
        
        # Reshape and extract 4 control points (4x2 matrix)
        points = ca.reshape(ctrl_points, (self.num_ctrl_points, self.m))
        segment = ca.vertcat(
            points[seg_idx, :],     # First control point
            points[seg_idx+1, :],   # Second control point
            points[seg_idx+2, :],   # Third control point
            points[seg_idx+3, :]    # Fourth control point
        )
        
        # Convert to 4x2 matrix explicitly
        segment_reshaped = ca.reshape(segment, 4, self.m)
        
        # Matrix multiplication: (1x4) @ (4x4) @ (4x2) = (1x2)
        return ca.mtimes(ca.mtimes(basis.T, self.M), segment_reshaped).T

    def controls_to_points(self, U):
        """Convert control sequence to B-spline control points"""
        U_flat = U.reshape(-1, self.m)
        return np.linalg.lstsq(self.basis_matrix, U_flat, rcond=None)[0]

    def points_to_controls(self, q):
        """Convert B-spline control points to control sequence"""
        return self.basis_matrix @ q.reshape(self.num_ctrl_points, self.m)
    

# Create B-spline generator: 4 control points -> 20 controls over 5 seconds
bs = BSplineGenerator(
    degree=3,
    num_ctrl_points=4,
    time_horizon=5.0,
    control_dim=2,
    num_samples=20
)

# Convert between representations
q = np.random.randn(4, 2)  # Random control points
U = bs.points_to_controls(q)  # Generate 20 controls
print(U.shape)  # Expected: (20, 2)
q_recovered = bs.controls_to_points(U)  # Project back to control points

# Symbolic evaluation at specific time
t = ca.MX.sym('t')
q_sym = ca.MX.sym('q', 4*2)
u = bs.symbolic_control(t, q_sym)  # CasADi symbolic function
print(u.shape)  # Expected: (2, 1)