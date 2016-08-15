#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pylab as pl

from sympy.abc import x
from scipy.integrate import quad
from scipy.optimize import basinhopping
from scipy.linalg import expm
from scipy.linalg import logm


from math import fabs
from scipy.integrate import quad
from geometry_msgs.msg import Pose2D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from random import randint
from mpl_toolkits.mplot3d import Axes3D

# import usefull classes
from parameters import *
from simu_classes import *
from model import *
from MPC import *



def simulate(u, current_state) :

    global integration_factor
    global horizon_iteration

    global Vx

    global T_discretisation

    global psi_range
    global Ad_range
    global Bd_range

    global exponential_factor
    global integration_factor
    

    T = T_discretisation
    lim = len(u)

    X = [current_state.x]
    Y = [current_state.y]
    
    #b = get_model_b_matrix()
    
    x_i = sp.Matrix([current_state.Vy, current_state.Vpsi, current_state.psi],dtype=float_)
    u_i = sp.Matrix(u[0],dtype=float_)

    print "x_0 : ",x_i
    print "u_0 : ",u_i

    a = get_model_a_matrix()
    A = np.matrix([[a[0],a[1],0],[a[2],a[3],0],[0,1,0]])
    b = get_model_b_matrix()
    B   = np.matrix([[b[0],b[1]],[b[4],b[5]],[0,0]])
    x   = sp.Symbol("x")

    Ad = expm(A*T)
    Bd = integrate_m(fast_expm(sp.Matrix(A)*x,3,exponential_factor)*B,3,2,0,T,T/integration_factor)

    print "Ad : "
    print Ad
    print "Bd : "
    print Bd

    for i in range(1,lim):
        
        # calcul de l'état à l'itération suivante
        if not type(Ad) == type(0) and not type(Bd) == type(0) :
            
            x_i  = Ad*x_i + Bd*u_i

            if i<lim :
                u_i = sp.Matrix(u[i],dtype=float_)

            psi = float(x_i[2])
            Vy  = float(x_i[0])

            X.append(X[-1] + Vx*np.cos(psi)*T - Vy*np.sin(psi)*T)
            Y.append(Y[-1] + Vx*np.sin(psi)*T + Vy*np.cos(psi)*T)
        else :
            print "[X] Error model calculation"

    return X, Y



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
    X, Y = simulate(u, current_state)

    #display_states(states)

    # affichage des trajectoires obtenues
    plt.figure("Evolution de la trajectoire du robot")
    plt.plot(X,Y)
    plt.show()