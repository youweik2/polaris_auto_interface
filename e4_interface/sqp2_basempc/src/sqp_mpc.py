#!/usr/bin/env python3

# Basic
import numpy as np
from scipy import signal
from math import *
# import alvinxy.alvinxy as axy
import rospy

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
from akm_mpc.BaseMPC import GemCarOptimizer
from akm_mpc.BaseMPC import GemCarModel

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
        self.omega      = 0.0

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
        self.circle_obstacles_1 = {'x': 0.3, 'y': 20, 'r': 2.0}
        self.circle_obstacles_2 = {'x': 0, 'y': 10, 'r': 2.0}

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

        self.curr_x = local_x_curr - self.offset * np.cos(self.curr_yaw)
        self.curr_y = local_y_curr - self.offset * np.sin(self.curr_yaw)

    def start_loop(self):

        # Car model setting
        self.obstacles = np.array([
                [0.3, 20, 2],        #x, y, r 20 25 30
                # [0.0, 10, 2],
                # [-0.5, 30, 2],
                ])
        car_model = GemCarModel()
        self.opt = GemCarOptimizer(m_model=car_model.model, 
                                m_constraint=car_model.constraint, t_horizon=self.horizon, dt=self.dt, obstacles = self.obstacles, target=self.target)
        
        
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

                    self.gem_enable = True

                    self.rate.sleep()

                    x_init, y_init, theta_init, vel_init = self.curr_x, self.curr_y, self.angle, 0.0

                    self.x_log.append(x_init)
                    self.y_log.append(y_init)
                    self.theta_log.append(theta_init)
                    self.v_log.append(vel_init)

            # Cord command
            cord = Pose2D()
            self.get_gem_state()
            cord.x = self.curr_x
            cord.y = self.curr_y
            cord.theta = self.curr_yaw
            self.cord_pub.publish(cord)

            self.mpc_interface(self.curr_x, self.curr_y, self.angle, self.speed, self.fai, self.acc, self.omega)
            self.rate.sleep()

        self.x_log.pop()
        self.y_log.pop()
        self.theta_log.pop()
        self.v_log.pop()

        self.plot_results(self.x_log, self.y_log, self.theta_log, self.v_log, self.x_real_log, self.y_real_log, self.theta_real_log, self.v_real_log, self.a_log, self.steer_log)
        
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

    def mpc_interface(self, x_real, y_real, theta_real, vel_real, fai_real, a_real, o_real):

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

        # Solve !!!

        vel_real = min(2.99, vel_real)
        x_1, y_1, theta_1, fai_1, vel_1, a_1, o_1 = self.opt.solve(x_real, y_real, theta_real, fai_real, vel_real, a_real, o_real)

        self.acc, self.omega = a_1, o_1

        # Save & Filter
        # if vel_real < 3.5:
        #     self.acc = a_1 * 1.0 + a_real * 0.0 
        # else:
        #     self.acc = a_1 * 0.5
        #     print('brake')

        self.fai = fai_1 * 1.0
        print(self.fai)

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

        
    def plot_results(self, x_log, y_log, theta_log, v_log, x_real_log, y_real_log, theta_real_log, v_real_log, a_log, steer_log):

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
                                    self.circle_obstacles_2['r'], color='whitesmoke', fill=True))
        circles.append(plt.Circle((self.circle_obstacles_1['y'], self.circle_obstacles_1['x']),
                                    self.circle_obstacles_1['r'], color='k', fill=False))
        circles.append(plt.Circle((self.circle_obstacles_2['y'], self.circle_obstacles_2['x']),
                                    self.circle_obstacles_2['r'], color='k', fill=False))

        for circle in circles:
            axs2[2].add_artist(circle)
        
        axs2[2].invert_yaxis()
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