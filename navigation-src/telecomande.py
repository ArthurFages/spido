#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import rospy
import time
import numpy as np

from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64, Float32, Int32, String
from spido_pure_interface.msg import cmd_car

rospy.init_node('telecommande')

print "[*] Definition des publishers"
cmd_safe_publisher  = rospy.Publisher('cmd_car_safe',cmd_car,queue_size=10)
cmd_publisher       = rospy.Publisher('cmd_car',cmd_car,queue_size=10)
state_publisher     = rospy.Publisher('state',Int32,queue_size=10)
err_path_publisher	= rospy.Publisher('err_path',String,queue_size=10)

cmd = cmd_car()

cmd.linear_speed = 0
cmd.steering_angle_front = 0
cmd.steering_angle_rear = 0
cmd_publisher.publish(cmd)
cmd_safe_publisher.publish(cmd)
state_publisher.publish(1)

while True :
    commande = input(">")
    if commande == 8 :
        cmd.linear_speed += 1
    elif commande == 5 :
        cmd.linear_speed -= 1
    elif commande == 4 :
        cmd.steering_angle_front += np.radians(10)
        cmd.steering_angle_rear -= np.radians(10)
    elif commande == 6 :
        cmd.steering_angle_front -= np.radians(10)
        cmd.steering_angle_rear += np.radians(10)
    else :
        print "Apprend a taper bord's !"
    cmd_publisher.publish(cmd)
    cmd_safe_publisher.publish(cmd)
    state_publisher.publish(1)
