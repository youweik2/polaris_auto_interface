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
        self.wheelbase  = 2.565 # meters 2.565-e4/1.75-e2
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
            (0, 20, 1), (1, 25, 1), (-1, 30, 1)
        ]
        self.dt = 0.05
                # Obstacle set here
        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

        self.terminal = np.array([0.0, 40.0, np.pi/2, 0.0, 0.5])
        self.target_x = self.terminal[0]
        self.target_y = self.terminal[1]

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

        return -round(y,4), round(x,4), round(yaw + np.pi/2,4), round(vel,4)

    # plot function
    def plot_results(self, start_x, start_y, x_log, y_log, theta_log, v_log, x_real_log, y_real_log, theta_real_log, v_real_log, a_log, steer_log):

        timeline = np.arange(0, (len(a_log)), 1)*self.dt

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        axs[0].plot(timeline, a_log, 'r-', label='accel')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('value')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(timeline, steer_log, 'r-', label='steer angle')
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('value')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(timeline, y_real_log, 'r-', label='y move')
        axs[2].set_xlabel('time')
        axs[2].set_ylabel('value')
        axs[2].legend()
        axs[2].grid(True)
        plt.tight_layout()
        plt.show()

        fig, axs2 = plt.subplots(3, 1, figsize=(12, 18))

        axs2[0].plot(timeline, theta_log, 'r-', label='desired theta')
        axs2[0].plot(timeline, theta_real_log, 'b-', label='real theta')
        axs2[0].set_xlabel('Time')
        axs2[0].set_ylabel('Theta')
        axs2[0].legend()
        axs2[0].grid(True)


        axs2[1].plot(timeline, v_log, 'r-', label='desired velocity')
        axs2[1].plot(timeline, v_real_log, 'b-', label='real velocity')
        axs2[1].set_xlabel('Time')
        axs2[1].set_ylabel('Velocity')
        axs2[1].legend()
        axs2[1].grid(True)

        axs2[2].plot(y_log, x_log, 'r-', label='desired path')
        axs2[2].plot(y_real_log, x_real_log, color='b', linestyle='--', label='real path')
        axs2[2].plot(self.target_y, self.target_x, 'bo', label='target')
        # axs2[2].plot(start_y, start_x, 'go', label='start')
        axs2[2].set_xlabel('Pos Y')
        axs2[2].set_ylabel('Pos X')

        circles = []
        circles.append(plt.Circle((self.circle_obstacles_1['y'], self.circle_obstacles_1['x']),
                                    self.circle_obstacles_1['r'], color='whitesmoke', fill=True))
        circles.append(plt.Circle((self.circle_obstacles_2['y'], self.circle_obstacles_2['x']),
                                    self.circle_obstacles_2['y'], color='whitesmoke', fill=True))
        circles.append(plt.Circle((self.circle_obstacles_3['y'], self.circle_obstacles_3['x']),
                                    self.circle_obstacles_3['r'], color='whitesmoke', fill=True))
        circles.append(plt.Circle((self.circle_obstacles_1['y'], self.circle_obstacles_1['x']),
                                    self.circle_obstacles_1['r'], color='k', fill=False))
        circles.append(plt.Circle((self.circle_obstacles_2['y'], self.circle_obstacles_2['x']),
                                    self.circle_obstacles_2['r'], color='k', fill=False))
        circles.append(plt.Circle((self.circle_obstacles_3['y'], self.circle_obstacles_3['x']),
                                    self.circle_obstacles_3['r'], color='k', fill=False))

        for circle in circles:
            axs2[2].add_artist(circle)
            
        # axs2[2].invert_yaxis()
        axs2[2].axis('equal')
        axs2[2].legend()
        axs2[2].grid(True)
        plt.tight_layout()
        plt.show()

    def start_loop(self):

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
            for ox, oy, r in self.obstacles:
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

        bs = BSplineGenerator(
            degree=5,
            num_ctrl_points=8,
            time_horizon=10.0,
            control_dim=2,
            num_samples=100
        )
        
        ddp = BsplineDDP(
            Nx=5, Nu=2,
            dynamics=f,
            inst_cost=lambda x, u, x_goal: L(x, u, x_goal),
            terminal_cost=Phi,
            bspline_config=bs
        )
    
        x_current = ca.DM([0, 0, np.pi/2, 0, 0])
        x_goal = ca.DM([-0.0, 40, np.pi/2, 0, 0])
        q = np.zeros((bs.num_ctrl_points, 2))
        
        X_hist = [x_current.full().flatten()]
        U_hist = []

        a_old, phi_old = 0, 0
        

        x_0, y_0, theta_0,  curr_fai, vel_0, current_a, current_o = 0, 0, np.pi/2, 0.000, 0, 0, 0
        
        x_log, y_log, theta_log, v_log = [x_0], [y_0], [theta_0], [vel_0]
        x_real_log, y_real_log, theta_real_log, v_real_log = [], [], [], []
        steer_log, a_log = [curr_fai], [current_a]
        

        while not rospy.is_shutdown():

                        # get current position and orientation in the world frame
            curr_x, curr_y, curr_yaw, curr_vel = self.get_gem_pose()
            x_current = ca.DM([curr_x, curr_y, curr_yaw, curr_fai, curr_vel])

            if curr_y > 40 or ((curr_y > 35 or curr_y < -20) and curr_vel < 0.05):

                self.ackermann_msg.speed = 0
                x_log.pop()
                y_log.pop()
                theta_log.pop()
                v_log.pop()
                a_log.pop()
                steer_log.pop()

                self.plot_results(0, 0, x_log, y_log, theta_log, v_log, x_real_log, y_real_log, theta_real_log, v_real_log, a_log, steer_log)
                # break
                rospy.loginfo("Stopping the node...")
                rospy.signal_shutdown("Reach the target")


            # print(curr_x, curr_y, curr_yaw, curr_fai, curr_vel)

            X_opt, U_opt, q = ddp.optimize(
                x_current, x_goal, q,
                control_bounds={'a_min': -2.0, 'a_max': 0.8, 'fai_min': -np.pi/6, 'fai_max': np.pi/6},
                full_output=False
            )

            u_apply = U_opt[0]
            u_apply1 = U_opt[1]

            u_apply = np.clip(u_apply, [-2.0, -3], [0.8, 3])
            u_apply1 = np.clip(u_apply1, [-2.0, -3], [0.8, 3])
            
            x_current[2] = wrap_to_pi(x_current[2])

            x_current = f(x_current, ca.DM(u_apply))
            x_current = x_current.full().flatten()

            # Shift control points
            q = np.roll(q, -1, axis=0)
            q[-1] = q[-2]  # Maintain continuity

            X_hist.append(x_current)#.full().flatten())
            U_hist.append(u_apply)

            a_new = u_apply[0] # a_old * 0.5 + (u_apply[0] * 0.5 + u_apply1[0] * 0.5) * 0.5
            a_old = a_new

            curr_fai = curr_fai + (phi_old * 0.5 + (u_apply[1] * 0.5 + u_apply1[1] * 0.5) * 0.5) * 0.1
            phi_old = curr_fai

            curr_fai =  x_current[3] * 1.0 + phi_old * 0.00 #  curr_fai + u_apply[1] * 0.05
            steer_angle = curr_fai  # x_current[3] # self.front2steer(delta)
            speed_cmd =curr_vel + a_new * 0.1 #  x_current[3] 
            self.ackermann_msg.speed = speed_cmd
            self.ackermann_msg.steering_angle = steer_angle
            
            print(steer_angle, speed_cmd)

            # print(speed_cmd, steer_angle)

            self.ackermann_pub.publish(self.ackermann_msg)

            x_log.append(x_current[0])
            y_log.append(x_current[1])
            theta_log.append(x_current[2])
            v_log.append(x_current[4])

            x_real_log.append(curr_x)
            y_real_log.append(curr_y)
            theta_real_log.append(curr_yaw)
            v_real_log.append(curr_vel)

            a_log.append(a_new)
            steer_log.append(curr_fai * 0.8 + curr_fai * 0.2)

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
