#!/usr/bin/env python3

# Basic
import numpy as np
from scipy import signal
from math import *
# import alvinxy.alvinxy as axy
import rospy
import casadi as ca

# Message
from std_msgs.msg import String, Bool, Float32, Float64
from geometry_msgs.msg import Pose2D

# GEM Sensor Headers
from septentrio_gnss_driver.msg import INSNavGeod
from sensor_msgs.msg import NavSatFix

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

# MPC import
import pickle
import matplotlib.pyplot as plt
from bspline_ddp.bspline import BSplineGenerator
from bspline_ddp.ddp_bspline import BsplineDDP


def mdeglat(lat):
    latrad = lat*2.0*pi/360.0
    dy = 111132.09 - 566.05 * cos(2.0*latrad) \
         + 1.20 * cos(4.0*latrad) \
         - 0.002 * cos(6.0*latrad)
    return dy

def mdeglon(lat):
    latrad = lat*2.0*pi/360.0 
    dx = 111415.13 * cos(latrad) \
         - 94.55 * cos(3.0*latrad) \
	+ 0.12 * cos(5.0*latrad)
    return dx

def ll2xy(lat, lon, orglat, orglon):
    x = (lon - orglon) * mdeglon(orglat)
    y = (lat - orglat) * mdeglat(orglat)
    return (x,y)

class VehicleController():
    def __init__(self):

        self.bs = BSplineGenerator(
            degree=3,
            num_ctrl_points=8,
            time_horizon=3.0,
            control_dim=2,
            num_samples=60
        )

        self.rate       = rospy.Rate(20)
        self.look_ahead = 4
        self.wheelbase  = 2.565 # meters
        self.offset     = 1.26 # meters
        self.target     = [0, 40, np.pi/2, 0, 0]

        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0
        self.speed      = 0.0
        self.fai        = 0.0
        self.acc        = 0.0

        self.olat       = 40.092812    
        self.olon       = -88.236095

        self.gnss_sub   = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub    = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)
        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 3.5 # radians/second

        # MPC setting
        self.Epi = 3000
        self.plot_figures = True
        self.target_x = self.target[0]
        self.target_y = self.target[1]
        self.horizon = 2.0
        self.dt = 0.05
        self.obstacles = [
            (1.2, 15, 1.5) # , (-1.0, 30, 1.2) 
        ]

        self.circle_obstacles_1 = {'x': 0.5, 'y': 15, 'r': 1.5}
        self.circle_obstacles_2 = {'x': -1.0, 'y': 30, 'r': 1.2}

        self.u_old = np.array([1.0, 0.0])
        self.q = np.ones((self.bs.num_ctrl_points, 2))

        # publish point info
        self.cord_pub = rospy.Publisher('cord', Pose2D,  queue_size=1)  
        self.kal_cord_pub = rospy.Publisher('kal_cord', Pose2D, queue_size=1)

        # Desired control values
        self.steering_angle = 0.0  # Steering wheel angle in radians
        self.steering_speed_limit = 2 # Steering wheel rotation speed in radians/sec
        self.brake_percent = 0.0    # Brake command (0.0 to 1.0)
        self.throttle_percent = 0.0 # Throttle command (0.0 to 1.0)

        # initial params
        self.angle = 0.0
        self.steer_angle = 0.0
        self.delta = 0.0
        self.throttle_percent, self.brake_percent = 0.0, 0.0
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0

        # log info -- used for plotting
        self.x_log, self.y_log, self.theta_log, self.v_log = [], [], [], []
        self.x_real_log, self.y_real_log, self.theta_real_log, self.v_real_log = [], [], [], []
        self.steer_log, self.a_log = [], []


    def wrap_to_pi(self, theta):
        wrapped_theta = theta - 2 * ca.pi * ca.floor((theta + ca.pi) / (2 * ca.pi))
        return wrapped_theta

    def f(self, x, u, constrain=True):
        dt = 0.05
        length = 2.565
        return ca.vertcat(
            x[0] + x[4] * ca.cos(x[2]) * dt,
            x[1] + x[4] * ca.sin(x[2]) * dt,
            x[2] + x[4] * ca.tan(x[3]) / length * dt,
            x[3] + u[1] * dt,
            x[4] + u[0] * dt
        )

    def Phi(self, x, x_goal):
        Qf = np.diag([100, 100, 30.0, 0.0, 0.1])
        return (x - x_goal).T @ Qf @ (x - x_goal)

    def L(self, x, u, x_goal):
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

    def ins_callback(self, msg):
        # heading: degrees
        self.heading = round(msg.heading, 6)
        # angle: radians
        self.angle = self.heading / 180.0 * np.pi

    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3)

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        # degrees to radians
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle
    
    def wps_to_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y

        lat_wp_y, lon_wp_x = ll2xy(lat_wp, lon_wp, self.olat, self.olon)

        return lon_wp_x, lat_wp_y

    def accel2ctrl(self, expected_acceleration):
        if expected_acceleration > 0.0:
            throttle_percent = (expected_acceleration+2.3501) / 7.3454
            brake_percent = 0.0
        else:
            throttle_percent = 0.0
            brake_percent = abs(expected_acceleration)/20
        throttle_percent = np.clip(throttle_percent, 0.0, 0.45)
        brake_percent = np.clip(brake_percent, 0.0, 0.2)

        return throttle_percent, brake_percent

    def get_gem_state(self):

        # vehicle x, y position (meters)
        local_x_curr, local_y_curr = self.wps_to_xy(self.lon, self.lat)

        self.curr_yaw = self.heading_to_yaw(self.heading) 

        self.curr_x = local_x_curr - self.offset * np.cos(self.angle)
        self.curr_y = local_y_curr - self.offset * np.sin(self.angle)

    def start_loop(self):

        self.ddp = BsplineDDP(
            Nx=5, Nu=2,
            dynamics=self.f,
            inst_cost=lambda x, u, x_goal: self.L(x, u, x_goal),
            terminal_cost=self.Phi,
            bspline_config=self.bs
        )

        # MPC parameters
        self.x_goal = ca.DM([-0.0, 50, np.pi/2, 0, 0])
        self.q = np.ones((self.bs.num_ctrl_points, 2))
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):
                if(self.pacmod_enable == True):

                    # ---------- enable PACMod ----------

                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")


                    x_init, y_init, theta_init, vel_init = self.curr_x, self.curr_y, self.angle, 0.0

                    self.x_log.append(x_init)
                    self.y_log.append(y_init)
                    self.theta_log.append(theta_init)
                    self.v_log.append(vel_init)

                    self.gem_enable = True

            # Cord command
            cord = Pose2D()
            self.get_gem_state()
            cord.x = self.curr_x
            cord.y = self.curr_y
            cord.theta = self.angle
            self.cord_pub.publish(cord)

            print('-------')

            self.mpc_interface(self.curr_x, self.curr_y, self.angle, self.fai, self.speed)
            self.rate.sleep()

        self.x_log.pop()
        self.y_log.pop()
        self.theta_log.pop()
        self.v_log.pop()

        self.plot_results(x_init, y_init, self.x_log, self.y_log, self.theta_log, self.v_log, self.x_real_log, self.y_real_log, self.theta_real_log, self.v_real_log, self.a_log, self.steer_log)
        
    def publish_commands(self):

        # if (self.delta <= 30 and self.delta >= -30):
        #     self.turn_cmd.ui16_cmd = 1
        # elif(self.delta > 30):
        #     self.turn_cmd.ui16_cmd = 2 # turn left
        # else:
        #     self.turn_cmd.ui16_cmd = 0 # turn right
            
        self.turn_cmd.ui16_cmd = 1 

        self.accel_cmd.f64_cmd = self.accel_percent
        self.brake_cmd.f64_cmd = self.brake_percent
        self.steer_cmd.angular_position = self.steer_angle
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.steer_pub.publish(self.steer_cmd)
        self.turn_pub.publish(self.turn_cmd)

    def mpc_interface(self, x_real, y_real, theta_real, fai_real, vel_real):


        # Terminal State: Stop iff reached
        if (x_real - self.target_x) ** 2 + (y_real - self.target_y) ** 2 < 1:
            # break
            self.brake_cmd.f64_cmd = 0.7
            while (self.speed > 0.05):
                self.brake_pub.publish(self.brake_cmd)
            rospy.loginfo("Stopping the node...")
            rospy.signal_shutdown("Reach the target")
        # boundary condition
        if x_real < -10 or x_real > 10 or y_real > 45 or y_real < -50:
            self.brake_cmd.f64_cmd = 0.7
            while (self.speed > 0.05):
                self.brake_pub.publish(self.brake_cmd)
                
            # break
            rospy.loginfo("Stopping the node...")
            rospy.signal_shutdown("Exceed the bounds")


        x_current = ca.DM([x_real, y_real, theta_real, fai_real, vel_real])

        X_opt, U_opt, q1 = self.ddp.optimize(
            x_current, self.x_goal, self.q,
            control_bounds={'a_min': -2.0, 'a_max': 0.8,
                           'fai_min': -np.pi/5.2, 'fai_max': np.pi/5.2},
            full_output=False
        )
        
        u_apply = U_opt[0]
        # x_desire = X_opt[1]
        

        # Clip control inputs
        u_apply[0] = np.clip(u_apply[0], -2.0, 0.8)
        u_apply[1] = np.clip(u_apply[1], -np.pi/6, np.pi/6)

        # Clip control based on the continuous change
        # u_apply[1] = np.clip(u_apply[1], self.u_old[1]-0.3, self.u_old[1]+0.3)
        # u_apply[0] = np.clip(u_apply[0], self.u_old[0]-1, self.u_old[0]+1)
        # self.u_old = u_apply

        x_current[2] = self.wrap_to_pi(x_current[2])

        x_desire = self.f(x_current, ca.DM(u_apply)).full().flatten()


        x_1, y_1, theta_1, fai_1, vel_1, a_1, o_1 = x_desire[0], x_desire[1], x_desire[2], x_desire[3], x_desire[4], u_apply[0], u_apply[1]

        print('pos', x_real, y_real, theta_real, fai_real, vel_real)

        print('pos_1', x_1, y_1, theta_1, fai_1, vel_1)


        
        # Shift control points
        self.q = np.roll(q1, -1, axis=0)
        self.q[-1] = self.q[-2]  # Maintain continuity

        # Save & Filter
        self.acc = a_1 * 1.0 if vel_real < 3.5 else a_1 * 0.5
        self.fai = fai_1 * 1.0
        print(self.acc, self.fai, o_1, fai_real + o_1 * 0.05)

        # Trans to publish
        self.accel_percent, self.brake_percent = self.accel2ctrl(self.acc)
        
        self.delta = np.degrees(round(np.clip(self.fai, -0.61, 0.61), 3))
        self.steer_angle = -np.radians(self.front2steer(self.delta))

        self.publish_commands()

        # Plot log
        self.x_log.append(x_1)
        self.y_log.append(y_1)
        self.theta_log.append(theta_1)
        self.v_log.append(vel_1)

        self.x_real_log.append(x_real)
        self.y_real_log.append(y_real)        
        self.theta_real_log.append(theta_real)        
        self.v_real_log.append(vel_real) 

        self.a_log.append(self.acc)
        self.steer_log.append(self.steer_angle)

        
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
        axs2[2].plot(start_y, start_x, 'go', label='start')
        axs2[2].set_xlabel('Pos Y')
        axs2[2].set_ylabel('Pos X')
        

        circles = []
        circles.append(plt.Circle((self.circle_obstacles_1['y'], self.circle_obstacles_1['x']),
                                    self.circle_obstacles_1['r'], color='whitesmoke', fill=True))
        circles.append(plt.Circle((self.circle_obstacles_2['y'], self.circle_obstacles_2['x']),
                                    self.circle_obstacles_2['r'], color='whitesmoke', fill=True))
        circles.append(plt.Circle((self.circle_obstacles_1['y'], self.circle_obstacles_1['x']),
                                    self.circle_obstacles_1['r'], color='k', fill=False))
        circles.append(plt.Circle((self.circle_obstacles_2['y'], self.circle_obstacles_2['x']),
                                    self.circle_obstacles_2['r'], color='k', fill=False))

        for circle in circles:
            axs2[2].add_artist(circle)

        axs2[2].axis('equal')
        axs2[2].legend()
        axs2[2].grid(True)
        plt.tight_layout()
        plt.show()

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