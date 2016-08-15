#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import rospy
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from pylab import *
import time
import sympy

from sympy.abc import x
from scipy.integrate import quad
from scipy.optimize import basinhopping
from scipy.linalg import expm
from scipy.linalg import logm
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
from model import *
from MPC import *

def fact(n):
    """fact(n): calcule la factorielle de n (entier >= 0)"""
    if n<2:
        return 1
    else:
        return n*fact(n-1)

def fast_expm(A,lenA,k) :
    result = sympy.zeros(lenA)
    for i in range(0,k) :
        result += (1/fact(i))*(A**i)
    return result

"""
Intègre tout les coefficients d'une matrice
"""
def integrate_m(M,len_lin,len_col,t0,tf) :
    x = sympy.Symbol("x")
    result = sympy.zeros(len_lin,len_col)
    for i in range(0,len_lin) :
        for j in range(0,len_col) :
            #print sympy.Integral(M[i,j],(x,t0,tf))
            result[i,j] = sympy.integrate(M[i,j],(x,t0,tf))
    return result


def fast_integration(f,t0,tf,dt) :
    
    result  = 0
    t       = t0
    
    while t<tf :
        result = result + abs(f(t)*dt)
        t = t + dt
    
    return result

def simulate(u, current_state) :

    global integration_factor
    global horizon_iteration

    global Vx

    global T_discretisation

    global psi_range
    global Ad_range
    global Bd_range
    

    T = T_discretisation
    print "T simulate : ",T
    lim = len(u)

    X = [current_state.x]
    Y = [current_state.y]
    
    #b = get_model_b_matrix()
    
    x_i = sympy.Matrix([current_state.Vpsi, current_state.Vx, current_state.Vy, current_state.psi, current_state.x, current_state.y],dtype=float_)
    u_i = sympy.Matrix(u[0],dtype=float_)

    print "x_0 : ",x_i
    print "u_0 : ",u_i
    
    states = [current_state]

    Ad = get_Ad(states[-1],T)
    Bd = get_Bd(states[-1],T)

    print "Ad : "
    print Ad
    print "Bd : "
    print Bd

    for i in range(1,lim):

        #print "states[-1] : ",states[-1]

        Ad = get_Ad(states[-1],T)
        Bd = get_Bd(states[-1],T)
        
        # calcul de l'état à l'itération suivante
        if not type(Ad) == type(0) and not type(Bd) == type(0) :
            
            x_i  = Ad*x_i + Bd*u_i

            if i<lim :
                u_i = sympy.Matrix(u[i],dtype=float_)

            states.append(State(  x_i[0], 
                                    x_i[1],
                                    x_i[2], 
                                    x_i[3], 
                                    x_i[4], 
                                    x_i[5],
                                    u_i[0],
                                    u_i[1], 
                                    current_state.t+i*T))
            print states[-1].to_string_v()

            X.append(x_i[4])
            Y.append(x_i[5])
        else :
            print "[X] Error model calculation"
                                

    return X, Y, states


def break_point() :
    exit(0)

def display_states(states) :

    Vx = []
    Vy = []
    Vpsi = []
    psi = []
    x = []
    y = []
    bf = []
    br = []

    for state in states :
        Vx.append(state.Vx)
        Vy.append(state.Vy)
        Vpsi.append(state.Vpsi)
        psi.append(state.psi)
        y.append(state.y)
        x.append(state.x)
        bf.append(state.beta_f)
        br.append(state.beta_r)

    plt.figure("Evolution etat référentiel du SPIDO")
    plt.subplot(2,2,1)
    plt.title("x au cours du temps")
    plt.plot(range(0,len(x)),x)
    plt.subplot(2,2,2)
    plt.title("y au cours du temps")
    plt.plot(range(0,len(y)),y)
    plt.subplot(2,2,3)
    plt.title("Psi au cours du temps")
    plt.plot(range(0,len(psi)),psi)
    plt.subplot(2,2,4)
    plt.title("Vpsi au cours du temps")
    plt.plot(range(0,len(Vpsi)),Vpsi)    
    
    plt.figure("Evolution etat référentiel du SPIDO - 2")
    plt.subplot(2,2,1)
    plt.title("Vx au cours du temps")
    plt.plot(range(0,len(Vx)),Vx)
    plt.subplot(2,2,2)
    plt.title("Vy au cours du temps")
    plt.plot(range(0,len(Vy)),Vy)
    plt.subplot(2,2,3)
    plt.title("Beta r au cours du temps")
    plt.plot(range(0,len(br)),br)
    plt.subplot(2,2,4)
    plt.title("Beta f au cours du temps")
    plt.plot(range(0,len(bf)),bf)


if __name__ == '__main__':
    
    global horizon_iteration # nombre de commandes prédites

    # détermination de la trajectoire de référence
    current_state   = State(0,4,0,np.radians(0),0,0.0,0,0,0)
    current_command = Command(0,0)

    # calcule les angles de braquages nécessaires au recalage du robot sur la trajectoire de référence
    # utilisation du MPC
    # détermination des prévisions de commande initial
    psi = np.radians(10)
    psi2 = np.radians(20)
    u = []
    for i in range(0,horizon_iteration) :
        u.append([-psi,0])#[current_command.betaf, current_command.betar]

    print "u : ",u

    # simulation de la rectification sur la trajectoire de référence
    #set_offline_model_range() # calcul offline du modèle linéarisé sous forme discrète
    X, Y, states = simulate(u, current_state)

    display_states(states)

    # affichage des trajectoires obtenues
    plt.figure("Evolution de la trajectoire du robot")
    plt.plot(X,Y)
    plt.show()