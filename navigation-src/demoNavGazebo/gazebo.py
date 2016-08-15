#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import rospy
import time

from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from parameters import *

# == Ressources partagées pour l'exécution parrallèle ============================

STATE = None # état du robot à un instant t
BETAF = 0
BETAR = 0

T0 = 0          # origine des temps
T = 0           # temps actuel

# repère de départ
X0 = 0
Y0 = 0
PSI0 = 122

# ====================================================================================

def get_topic_values() :
    global STATE, BETAR, BETAF, T
    return STATE, BETAR, BETAF, T


"""
Enregistrement du temps simulé
"""
def new_time(data):
    global T
    sec     = data.clock.secs   # secondes
    nsec    = data.clock.nsecs  # nanosecondes
    T = sec+nsec*(1e-9)            # conversion en secondes

"""
Enregistrement de l'état du SPIDO à un instant donné
"""
def new_position(data):
    
    global T
    global T0
    #global current_state
    #global states
    
    global PSI0
    global X0
    global Y0

    global STATE
    global BETAR 
    global BETAF
    
    dt = 0
    
    # enregistrement de quand on a fait la mesure
    TIME = T - T0

    #print "["+str(TIME)+"]"
    
    # récupération du torseur position
    X   = data.pose.pose.position.x
    Y   = data.pose.pose.position.y
    sin = data.pose.pose.orientation.z
    cos = data.pose.pose.orientation.w
    if sin < 0 :
        PSI = -2*np.arccos(cos)
    else :
        PSI = 2*np.arccos(cos)
        
    if PSI0 == 122 :
        # initialisation du repère de départ =/= repère global
        PSI0   = PSI
        X0     = X
        Y0     = Y
        
    # on considère les coordonées dans le repère de départ
    delta_x = np.sign(X)*(abs(X) - abs(X0)) 
    delta_y = np.sign(Y)*(abs(Y) - abs(Y0))
    X = delta_x*np.cos(PSI0) + delta_y*np.cos(np.radians(90) - PSI0)
    Y = delta_y*np.cos(PSI0) - delta_x*np.cos(np.radians(90) - PSI0)
    PSI = PSI - PSI0
    
    # calcul des vitesses dans le repère global
    VX_abs = data.twist.twist.linear.x
    VY_abs = data.twist.twist.linear.y
    
    # calcul des vitesses dans le repère lié au robot
    VPSI = data.twist.twist.angular.z
    VX = VX_abs*np.cos(PSI) + VY_abs*np.cos(np.radians(90)-PSI)
    VY = VY_abs*np.cos(PSI) - VX_abs*np.cos(np.radians(90)-PSI)

    STATE = State(VPSI, VX, VY, PSI, X, Y, BETAF, BETAR, TIME)
    #states.append(STATE)


def set_rospy_listener() :

    global T0

    # réception de l'état réel du robot (position+temps)
    rospy.Subscriber("odom", Odometry, new_position)
    rospy.Subscriber("clock", Clock, new_time)
    T0 = T
    time.sleep(2)
