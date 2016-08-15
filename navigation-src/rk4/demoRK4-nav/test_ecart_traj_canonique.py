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
from classes import *
from model import *
from MPC import *
from trajectory import *

def simulate(u, current_state) :

    global sum_factor
    global integration_factor

    global horizon_iteration
    global thau_rk

    global Vx

    global T_discretisation

    print "[simulate] U : ",u
    

    #T = tf/horizon_iteration
    #print "T simulate : ",T
    lim = len(u)
    t0 = 0

    X = [current_state.x]
    Y = [current_state.y]
    
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
        states.append(State(x_i[1],Vx,x_i[0],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1],t0+T*i))
        print states[-1].to_string_v()

        syst_CI = np.array([x_i[0],x_i[1],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1]]) 

        print "syst CI : ",syst_CI

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

def set_primitives(angles_braquage) :

    global horizon_iteration
    global Vx

    primitives = []
    id_man = 0

    # définition de l'état de référence à partir duquel sont calculés les pimitives
    current_state   = State(0,Vx,0,0,0,0,0,0,0)
    current_command = Command(0,0)

    for angle_braquage in angles_braquage :

        # réalisation commande
        u = [[current_command.betaf, current_command.betar]]
        for i in range(0,horizon_iteration-1) :
            u.append([angle_braquage,angle_braquage])

        # génération des états constituant la primitive de mouvement
        X,Y,states = simulate(u,current_state)

        # enregistrement de la primitive
        primitives.append(Maneuver( id_man, 
                                    X[-1] - X[0], 
                                    Y[-1] - Y[0], 
                                    states[-1].psi - states[0].psi,
                                    states[-1].t - states[0].t,
                                    0,0,'',states))
        # TODO : vérifier que la copie des états se fait bien en profondeur
        id_man += 1

    return primitives

def test_simulate() :

    angle_braquage = np.radians(10)

    current_state   = State(0,Vx,0,0,0,0,0,0,0)
    current_command = Command(0,0)
    u = [[current_command.betaf, current_command.betar]]
    for i in range(0,horizon_iteration-1) :
        u.append([angle_braquage,angle_braquage])

    X,Y,states = simulate(u,current_state)
    plt.plot(X,Y)
    plt.show()

def display_primitives(primitives) :
    
    for man in primitives :
        x = []
        y = []
        for state in man.states :
            x.append(state.x)
            y.append(state.y)
        plt.plot(x,y)
        
    plt.show()

def main() :
    
    global horizon_iteration # nombre de commandes prédites
    global betaMax
    global T_discretisation

    # définition des primitives de mouvement
    print "[*] Définition des primitives de mouvement"
    primitives = set_primitives([np.radians(-20),np.radians(-10),0,np.radians(10),np.radians(20)])
    display_primitives(primitives) # affichage de vérification

    # définition de la trajectoire de référence
    initial_state   = Output(0,0,0,0,0)
    target_state    = trajectory[0]

    # calcul de la trajectoire d'évitement d'obstacle par la méthode des champs de potentiel
    print "[*] Détermination trajectoire par méthode des champs de potentiels"
    pot_field_trajectory = set_potential_field_trajectory(initial_state,target_state)

    # lissage de la trajectoire par champs de potentiel avec les primitives de mouvement
    print "[*] Lissage de la trajectoire par les primitives de mouvement"
    seq_primitives = approximate(initial_state)

    # détermination de la trajectoire de référence
    #print "[*] Définition d'une suite de primitives de mouvement"
    #seq_fin = [maneuvers[0].copy()]

    print "[*] Définition des trajectoires de référence"
    ref_starting_state = State(0,4,0,0,0,0,0,0,0)
    states_ref = set_state_ref2(ref_starting_state,seq_primitives)
    
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
    
    # une commande = un angle de rotation = angle de braquage avant ET arrière
    u_0 = [0]*(horizon_iteration-1)
    u_0 = [u_0]

    # etablissement des contraintes
    cons = []
    for i in range(0,len(u_0[0])) :
        cons.append((-betaMax,betaMax))

    # détermination de la commande optimale par MPC
    t1 = time.time()
    u_optimized = minimize(fun=f_cost_MPC,x0=u_0, bounds=cons, options={'maxiter': 1},args=[current_state,current_command,u_ref,x_ref,t_ref]).x
    t2 = time.time()

    print "u optimized      : ",u_optimized
    print "computation time : ",(t2-t1)

    # mise en forme de u optimized
    u = [[current_state.beta_f,current_state.beta_r]]
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

if __name__ == '__main__':
    test_simulate()