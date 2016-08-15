#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import rospy
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from pylab import *
import time

from scipy.optimize import basinhopping
from scipy.linalg import expm3
from pyOpt import *

from math import fabs
from scipy.integrate import quad
from geometry_msgs.msg import Pose2D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from random import randint
from mpl_toolkits.mplot3d import Axes3D


#### GAZEBO SIMULATION

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32, String
from std_msgs.msg import Float64
from spido_pure_interface.msg import cmd_car
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock


# import usefull classes
from parameters import *
from simu_classes import *
from initial_simulation_setup import *
from MPC import *

def simulate(u, current_state) :

    global sum_factor
    global integration_factor

    global horizon_iteration
    global tf
    

    T = tf/horizon_iteration
    print "T simulate : ",T
    print "Limite de simulation : ",len(u)
    lim = len(u)

    X = [current_state.x]
    Y = [current_state.y]
    
    b = get_model_b_matrix()
    
    x_i = np.array([current_state.Vpsi, current_state.Vx, current_state.Vy, current_state.psi, current_state.x, current_state.y],dtype=float_)
    u_i = np.array(u[0],dtype=float_)
    print "x_0 : ",x_i
    print "u_0 : ",u_i

    states = []
    

    # len-t_ref) = horizon_iteration ou longueur maximale de la prédiction dans le cas ou on prédit au dela des trajectoires enregistrées
    for i in range(1,lim+1):
        
        # détermination du modèle au point de fonctionnement courant
        A   = get_model_a2_matrix_lin(current_state)
        Ad  = expm3(A*T)
        B   = np.array([[b[4],b[5]],[0,0],[b[0],b[1]],[0,0],[0,0],[0,0]])
        f   = lambda t : expm3(A*t).dot(B)
        Bd  = fast_integration(f,0,T,T/integration_factor)
        
        # calcul de l'état à l'itération suivante
        if not type(Ad) == type(0) and not type(Bd) == type(0) :
            
            x_i = Ad.dot(x_i) + Bd.dot(u_i)
            if i<lim :
                u_i = u[i]
            # self, Vpsi, Vx, Vy, psi, x, y, beta_f, beta_r, t=None) :
            states.append(State(    x_i[0], 
                                    x_i[1],
                                    x_i[2], 
                                    x_i[3], 
                                    x_i[4], 
                                    x_i[5],
                                    u_i[0],
                                    u_i[1], 
                                    current_state.t+i*T))
            X.append(x_i[4])
            Y.append(x_i[5])
        else :
            print "[X] Error model calculation"
                                

    return X, Y, states

def record_states(states, nom_primitive) :
    # enregistrement de la primitive dans un fichier au format csv
    
    fichier = open(nom_primitive,"a")
    
    for state in states :
        line = state.to_string()+"\n"
        fichier.write(line)
    
    fichier.close()

def make_left() :

    global horizon_iteration 

    angle_rotation = np.radians(-25)
    temps_simulation = 5

    # la commande lors d'un virage suit une fonction gaussienne
    m = temps_simulation/4       # "position de la gaussienne"
    sigma = temps_simulation   # "grosseur" gaussienne (real gaussian have curves)
    angle_braquage = lambda x: angle_rotation*np.exp((-1/2)*(((x-m)/sigma)**2))

    current_state = State(0,4,0,0,0,0,0,0,0)
    u = []
    for i in range(0,10) :
        u.append([0,0])
    for i in range(0,int(horizon_iteration)) :
        a = angle_braquage((temps_simulation/(horizon_iteration/2))*i)
        b = angle_braquage((temps_simulation/(horizon_iteration/2))*i)
        u.append([a,b])

    X, Y, states = simulate(u, current_state)

    plt.plot(X,Y)
    plt.show()

    # enregistrement de la primitive dans un fichier
    record_states(states,"virage_droite_model")

def make_s() :

    global horizon_iteration 

    angle_rotation = np.radians(-25)
    temps_simulation = 5
    current_state = State(0,4,0,0,0,0,0,0,0)


    # la commande lors d'un virage suit une fonction gaussienne
    m = temps_simulation/4       # "position de la gaussienne"
    sigma = temps_simulation/2   # "grosseur" gaussienne (real gaussian have curves)
    angle_braquage = lambda x: angle_rotation*np.exp((-1/2)*(((x-m)/sigma)**2))

    u = []
    for i in range(0,10) :
        u.append([0,0])
    for i in range(0,int(horizon_iteration/2)) :
        a = angle_braquage((temps_simulation/(horizon_iteration/2))*i)
        b = angle_braquage((temps_simulation/(horizon_iteration/2))*i)
        u.append([a,b])

    angle_rotation = np.radians(10)
    temps_simulation = 5

    # la commande lors d'un virage suit une fonction gaussienne
    m = temps_simulation/4       # "position de la gaussienne"
    sigma = temps_simulation/2   # "grosseur" gaussienne (real gaussian have curves)
    angle_braquage = lambda x: angle_rotation*np.exp((-1/2)*(((x-m)/sigma)**2))

    for i in range(0,10) :
        u.append([0,0])
    for i in range(0,int(horizon_iteration/2)) :
        a = angle_braquage((temps_simulation/(horizon_iteration/2))*i)
        b = angle_braquage((temps_simulation/(horizon_iteration/2))*i)
        u.append([a,b])

    print 'u : ',u

    X, Y, states = simulate(u, current_state)

    for state in states :
        print state.to_string_v()

    plt.plot(X,Y)
    plt.show()

    # enregistrement de la primitive dans un fichier
    record_states(states,"s")

if __name__ == '__main__':
    make_s()