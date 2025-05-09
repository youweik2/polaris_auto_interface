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
from akm_mpc.BaseMPC import GemCarOptimizer, GemCarModel

class VehicleController(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(20)

        self.look_ahead = 6    # meters
        self.wheelbase  = 2.565 # meters
        self.goal       = 0

        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        # Obstacle set here
        self.circle_obstacles_1 = {'x': 0, 'y': 15, 'r': 1}
        self.circle_obstacles_2 = {'x': 1, 'y': 20, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 25, 'r': 1.0}

        # MPC setting
        self.Epi = 3000
        self.plot_figures = True
        self.terminal = np.array([0.0, 40.0, np.pi/2, 0.0, 0.5])
        self.target_x = self.terminal[0]
        self.target_y = self.terminal[1]
        self.horizon = 2.0
        self.dt = 0.05
        self.obstacles = np.array([
                [-0.0, 15, 1],       #x, y, r 20 25 30
                [1.0, 20, 1],
                [-1.0, 25, 1]
                ])
        
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

        axs2[2].axis('equal')
        axs2[2].legend()
        axs2[2].grid(True)
        plt.tight_layout()
        plt.show()

    def start_loop(self):

        car_model = GemCarModel()
        self.opt = GemCarOptimizer(m_model=car_model.model, 
                                m_constraint=car_model.constraint, t_horizon=self.horizon, dt=self.dt, obstacles = self.obstacles, target=self.terminal)
        
        x_0, y_0, theta_0, vel_0, current_a, current_fai, current_o = -0.0, -0.0, np.pi/2, 0.0007, 0, 0, 0
        
        x_log, y_log, theta_log, v_log = [x_0], [y_0], [theta_0], [vel_0]
        x_real_log, y_real_log, theta_real_log, v_real_log = [], [], [], []
        steer_log, a_log = [current_a], [current_fai]


        while not rospy.is_shutdown():

            curr_x, curr_y, curr_yaw, curr_vel = self.get_gem_pose()
            

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

            x_1, y_1, theta_1, fai_1, vel_1, a_1, o_1 = self.opt.solve(curr_x, curr_y, curr_yaw, current_fai, curr_vel, current_a, current_o)


            current_fai, current_a, current_o = fai_1, a_1, max(min(o_1, 1), -1)


            delta = np.degrees(round(np.clip(fai_1, -0.61, 0.61), 3))

            steer_angle = delta 
            speed_cmd = curr_vel + a_1 * 0.05

            self.ackermann_msg.speed = speed_cmd
            self.ackermann_msg.acceleration = a_1
            self.ackermann_msg.steering_angle = np.radians(steer_angle)
            self.ackermann_pub.publish(self.ackermann_msg)

            x_log.append(x_1)
            y_log.append(y_1)
            theta_log.append(theta_1)
            v_log.append(vel_1)

            x_real_log.append(curr_x)
            y_real_log.append(curr_y)
            theta_real_log.append(curr_yaw)
            v_real_log.append(curr_vel)

            a_log.append(a_1)
            steer_log.append(fai_1)

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
