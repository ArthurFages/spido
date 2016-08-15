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
import eq_diff as eq


from sympy.abc import x
from scipy.integrate import quad
from scipy.optimize import basinhopping, fmin_cg, fmin_tnc, fmin_l_bfgs_b
from scipy.linalg import expm
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

    global sum_factor
    global integration_factor

    global horizon_iteration
    global thau_rk
    #global tf

    global Vx

    global T_discretisation

    print "[simulate] U : ",u
    

    #T = tf/horizon_iteration
    #print "T simulate : ",T
    lim = len(u)

    X = [current_state.x]
    Y = [current_state.y]
    
    b = get_model_b_matrix()
    stable_current_state = current_state.copy()
    
    # commande et états simulés à l'aide du modèle
    x_i     = np.array([stable_current_state.Vy, stable_current_state.Vpsi, stable_current_state.psi, stable_current_state.x, stable_current_state.y])
    u_i     = np.array(u[0])
    
    states = [current_state]

    print "states[-1] : ",states[-1]

    T = T_discretisation

    for i in range(1,len(u)):

        print "Commande : ",u_i

        # enregistement de l'état actuel
        states.append(State(x_i[1],Vx,x_i[0],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1]))
        #print states[-1].to_string_v()

        syst_CI = np.array([x_i[0],x_i[1],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1]]) 

        #print "syst CI : ",syst_CI

        # simulation du modèle
        t = pl.frange(0,T,thau_rk) 
        x_rk4    = eq.rk4(deriv, syst_CI, t)

        # enregistrement du nouvel etat
        for j in range(0,5) :
            x_i[j] = x_rk4[-1][j]

        u_i = u[i]
        
        # chargement de la commande prédite à l'itération suivante, et de la commande et trajectoire de référence à l'itération suivante
        X.append(x_i[3])
        Y.append(x_i[4])
                                

    return X, Y, states


def break_point() :
    exit(0)

if __name__ == '__main__':
    
    global horizon_iteration # nombre de commandes prédites
    global betaMax
    global T_discretisation
    #global states_ref

    #global current_state
    #global current_command
    
    #global u_ref
    #global x_ref
    #global t_ref

    #global seq_fin

    # détermination de la trajectoire de référence
    print "[*] Définition d'une suite de primitives de mouvement"
    seq_fin = [maneuvers[0].copy()]

    print "[*] Définition des trajectoires de référence"
    ref_starting_state = State(0,4,0,0,0,0,0,0,0)
    states_ref = set_state_ref2(ref_starting_state,seq_fin)
    
    Xref = []
    Yref = []
    for i in range(0,len(states_ref)) :
        Xref.append(states_ref[i].x)
        Yref.append(states_ref[i].y)

    # définition du chemin sur une horizon
    u_ref = get_u_ref(ref_starting_state, states_ref)
    x_ref = get_x_ref(ref_starting_state, states_ref)
    t_ref = get_t_ref(ref_starting_state, states_ref)

    # etats intial
    perturbation_y = 0.1
    orientation = np.radians(5)
    current_state   = State(0,4,0,orientation,0,perturbation_y,0,0,0)
    current_command = Command(0,0)

    # calcule les angles de braquages nécessaires au recalage du robot sur la trajectoire de référence
    # utilisation du MPC
    # détermination des prévisions de commande initial
    #u_0 = []
    #for i in range(0,horizon_iteration-1) :
    #    u_0 += [0,0]#[current_command.betaf, current_command.betar]
    
    # une commande = un angle de rotation = angle de braquage avant ET arrière
    u_0 = [0]*(horizon_iteration-1)
    u_0 = [u_0]

    # etablissement des contraintes
    cons = []
    for i in range(0,len(u_0[0])) :
        cons.append((-betaMax,betaMax))
    #cons = tuple(cons)
    
    #print "cons : ",cons

    t1 = time.time()
    

    Ad, Bd = get_discretized_model(current_state,T_discretisation)


    #u_optimized = basinhopping(func=f_cost_MPC, x0=u_0, minimizer_kwargs={'args':[current_state,current_command,u_ref,x_ref,t_ref,Ad,Bd]}, accept_test=mybounds, niter=1).x
    #u_optimized = minimize(fun=f_cost_MPC, x0=u_0, bounds=cons, options={'maxiter': 1},args=[current_state,current_command,u_ref,x_ref,t_ref,Ad,Bd]).x#,options={'maxiter':5}).x#, accept_test=mybounds, niter=1).x
    
    u_optimized = minimize(fun=f_cost_MPC,x0=u_0, bounds=cons, options={'maxiter': 1},args=[current_state,current_command,u_ref,x_ref,t_ref,Ad,Bd]).x#,options={'maxiter':5}).x#, accept_test=mybounds, niter=1).x
    
    #u_optimized = fmin_l_bfgs_b(func=f_cost_MPC, x0=u_0, bounds=cons, args=[[current_state,current_command,u_ref,x_ref,t_ref,Ad,Bd]]).x
    #u_optimized = fmin_tnc(func=f_cost_MPC, x0=u_0, bounds=cons, args=(current_state,current_command,u_ref,x_ref,t_ref)).x#,options={'maxiter':5}).x#, accept_test=mybounds, niter=1).x
    #u_optimized = minimize(fun=f_cost_MPC, x0=u_0, bounds=cons, method='L-BFGS-B', args=[current_state,current_command,u_ref,x_ref,t_ref]).x#,options={'maxiter':5}).x#, accept_test=mybounds, niter=1).x
    
    t2 = time.time()

    print "u optimized      : ",u_optimized
    print "computation time : ",(t2-t1)

    # mise en forme de u optimized
    u = [[current_state.beta_f,current_state.beta_r]]
    #for i in range(0,len(u_optimized),2) :
    #    u.append([u_optimized[i],u_optimized[i+1]])#[current_command.betaf, current_command.betar]

    for i in range(0,horizon_iteration-1) :
        u.append([u_optimized[i],u_optimized[i]])

    print "u : ",u


    # simulation de la rectification sur la trajectoire de référence
    X, Y, states = simulate(u, current_state)

    # calcul de l'erreur quadratique selon y
    delta_y = []
    for i in range(0,min(len(Yref),len(Y))) :
        delta_y.append((Yref[i]-Y[i])**2)
    print "quad error y simu :",delta_y

    # calcul de l'erreur quadratique moyenne selon y
    eqm = np.array(delta_y).mean()
    eqm_array_y = []
    for i in range(0,len(Yref)) :
        eqm_array_y.append(eqm)

    # calcul de l'erreur quadratique selon x
    delta_x = []
    for i in range(0,min(len(Xref),len(X))) :
        delta_x.append((Xref[i]-X[i])**2)
    print "quad error x simu :",delta_x

    # calcul de l'erreur quadratique moyenne selon x
    eqm = np.array(delta_x).mean()
    eqm_array_x = []
    for i in range(0,len(Xref)) :
        eqm_array_x.append(eqm)


    # affichage des trajectoires obtenues
    plt.figure("Evolution de la trajectoire du robot")
    plt.plot(Xref, Yref)
    plt.plot(X,Y)
    plt.show()

    # affichage des erreurs
    plt.figure("Erreur quadratique selon x")
    plt.plot(range(0,len(delta_x)), delta_x)
    plt.plot(range(0,len(eqm_array_x)), eqm_array_x)
    plt.show()

    plt.figure("Erreur quadratique selon y")
    plt.plot(range(0,len(delta_y)), delta_y)
    plt.plot(range(0,len(eqm_array_y)), eqm_array_y)
    plt.show()