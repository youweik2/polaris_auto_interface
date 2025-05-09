#!/usr/bin/env python3

#================================================================
# File name: pure_pursuit_sim.py                                                                  
# Description: pure pursuit controller for GEM vehicle in Gazebo                                                              
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 07/10/2021                                                                
# Date last modified: 07/15/2021                                                          
# Version: 0.1                                                                    
# Usage: rosrun gem_pure_pursuit_sim pure_pursuit_sim.py                                                                    
# Python version: 3.8                                                             
#================================================================

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import casadi as ca

# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Twist, Vector3
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Gazebo Headers
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState

# MPC Headers
from bspline_ddp.bspline import BSplineGenerator
from bspline_ddp.ddp_bspline import BsplineDDP

class VehicleController(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(20)

        self.look_ahead = 6    # meters
        self.wheelbase  = 1.75 # meters 2.565-e4/1.75-e2
        self.goal       = 0

        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        # MPC setting
        self.obstacles = [
            (5, 0, 0.5), (18, 1, 1), (32, 0, 1.5)
        ]


    def get_gem_pose(self):

        rospy.wait_for_service('/gazebo/get_model_state')
        
        try:
            service_response = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service_response(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: " + str(exc))

        x = model_state.pose.position.x
        y = model_state.pose.position.y

        orientation_q      = model_state.pose.orientation
        orientation_list   = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        v_world_x = model_state.twist.linear.x
        v_world_y = model_state.twist.linear.y
        vel = v_world_x * math.cos(yaw) + v_world_y * math.sin(yaw)

        return round(x,4), round(y,4), round(yaw,4), round(vel,4)

    def start_loop(self):

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
            for ox, oy, r in self.obstacles:
                dist = ca.sqrt((x[0]-ox)**2 + (x[1]-oy)**2)
                safe_dist = r + 0.8  # Increased safety margin
                
                # Gradient-aware penalty function
                penalty_scale = 1000.0  # Increased penalty magnitude
                steepness = 10.0  # Sharper transition
                obstacle_cost += penalty_scale * ca.exp(-steepness * (dist - safe_dist))
            
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


        bs = BSplineGenerator(
            degree=3,
            num_ctrl_points=8,
            time_horizon=3.0,
            control_dim=2,
            num_samples=60
        )

        ddp = BsplineDDP(
            Nx=3, Nu=2,
            dynamics=f,
            inst_cost=lambda x, u, x_goal: L(x, u, x_goal),
            terminal_cost=Phi,
            bspline_config=bs
        )

        x_current = ca.DM([0, 0, 0])  # X_INIT
        x_goal = ca.DM([50, -1, 0]) # X_TERMINAL
        q = np.ones((bs.num_ctrl_points, 2))
        
        X_hist = [x_current.full().flatten()]
        U_hist = []

        u_old = np.array([1.0, 0.0])
        

        while not rospy.is_shutdown():

            # get current position and orientation in the world frame
            curr_x, curr_y, curr_yaw, curr_vel = self.get_gem_pose()
            x_current = ca.DM([curr_x, curr_y, curr_yaw])

            X_opt, U_opt, q = ddp.optimize(
            x_current, x_goal, q,
            control_bounds={'v_min': -3.0, 'v_max': 3.0,
                           'omega_min': -0.5, 'omega_max': 0.5},
            full_output=False
            )

            u_apply = U_opt[0]
            u_apply[0] = np.clip(u_apply[0], -3.0, 3.0) # velocity
            u_apply[1] = np.clip(u_apply[1], -0.5, 0.5) # angular velocity

            u_apply[0] = np.clip(u_apply[0], u_old[0]-0.5, u_old[0]+0.5)
            u_apply[1] = np.clip(u_apply[1], u_old[1]-0.15, u_old[1]+0.15)
            u_old = u_apply

            # Shift control points
            q = np.roll(q, -1, axis=0)
            q[-1] = q[-2]  # Maintain continuity

            X_hist.append(x_current.full().flatten())
            U_hist.append(u_apply)

            steer_angle = np.arctan(self.wheelbase*u_apply[1]/curr_vel) # self.front2steer(delta)
            speed_cmd = u_apply[0]
            self.ackermann_msg.speed = speed_cmd
            self.ackermann_msg.steering_angle = steer_angle

            print(speed_cmd, steer_angle)

            self.ackermann_pub.publish(self.ackermann_msg)

            if (curr_x - 50) ** 2 + (curr_y - 0) ** 2 < 1 or (curr_x > 48.5 and curr_vel < 0.05):

                self.ackermann_msg.speed = 0

                    # Visualization
                X_hist = np.array(X_hist)
                U_hist = np.array(U_hist)
                
                fig, ax = plt.subplots(3, 1, figsize=(10, 12))
                
                # Trajectory plot
                ax[0].plot(X_hist[:,0], X_hist[:,1], 'b-', label='Trajectory')
                ax[0].plot(x_goal[0], x_goal[1], 'g*', markersize=15, label='Goal')
                for ox, oy, r in self.obstacles:
                    ax[0].add_patch(plt.Circle((ox, oy), r, color='r', alpha=0.3))
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

                # break
                rospy.loginfo("Stopping the node...")
                rospy.signal_shutdown("Reach the target")

            self.rate.sleep()
        

def main():
    rospy.init_node('mpc_node', anonymous=True)
    rospy.loginfo("MPC Node Start")
    controller = VehicleController()

    try:
        controller.start_loop()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
