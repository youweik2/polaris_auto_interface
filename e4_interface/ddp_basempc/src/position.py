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

        # initial params
        self.angle = 0.0
        self.steer_angle = 0.0
        self.delta = 0.0
        self.throttle_percent, self.brake_percent = 0.0, 0.0
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0



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

    def wps_to_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y

        lat_wp_y, lon_wp_x = ll2xy(lat_wp, lon_wp, self.olat, self.olon)

        return lon_wp_x, lat_wp_y

    def start_loop(self):

        
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

            # Cord command
            cord = Pose2D()
            self.get_gem_state()
            cord.x = self.curr_x
            cord.y = self.curr_y
            cord.theta = self.angle
            self.cord_pub.publish(cord)

            print('-------')
            print('x', self.curr_x)
            print('y', self.curr_y)

            self.mpc_interface(self.curr_x, self.curr_y, self.angle, self.fai, self.speed)
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