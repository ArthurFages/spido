#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import odespy
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

def simulate(u, current_state,T_discretisation) :

    global sum_factor
    global integration_factor

    #global horizon_iteration
    global thau_rk
    #global tf

    global Vx


    #print "[simulate] U : ",u
    

    #T = tf/horizon_iteration
    #print "T simulate : ",T
    lim = len(u)

    X = [current_state.x]
    Y = [current_state.y]
    
    #b = get_model_b_matrix()
    stable_current_state = current_state.copy()

    print "initial state ; ",stable_current_state.to_string_v()
    print "T discretisation ; ",T_discretisation
    print "horizon : ",horizon_iteration

    # commande et états simulés à l'aide du modèle
    x_i     = np.array([stable_current_state.Vy, stable_current_state.Vpsi, stable_current_state.psi, stable_current_state.x, stable_current_state.y])
    u_i     = np.array(u[0])
    
    states = [current_state]

    #print "states[-1] : ",states[-1]

    T = T_discretisation

    for i in range(1,len(u)):

        print "Commande : ",u_i

        # enregistement de l'état actuel
        states.append(State(x_i[1],Vx,x_i[0],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1]))
        #print states[-1].to_string_v()

        x_i = get_discret_following_state(x_i, u_i, T_discretisation)
        #x_i = get_ode_stiff_solution(x_i, u_i, T_discretisation)

        u_i = u[i]
        
        # chargement de la commande prédite à l'itération suivante, et de la commande et trajectoire de référence à l'itération suivante
        X.append(x_i[3])
        Y.append(x_i[4])

        print "Nouvel etat :",x_i

    print "Pas de discretisation : ",T_discretisation
                                

    return X, Y, states


def break_point() :
    exit(0)

def get_rk4_solution(current_state_tab, command, T_discretisation) :

    global thau_rk

    # conditions initiales
    syst_CI = np.array([current_state_tab[0],current_state_tab[1],current_state_tab[2],current_state_tab[3],current_state_tab[4],command[0],command[1]])
    
    # simulation du modèle
    t = pl.frange(0,T_discretisation,thau_rk) 
    x_rk4    = eq.rk4(deriv, syst_CI, t)

    return np.array(x_rk4[-1])

def get_matrix_solution(current_state_tab, command, T_discretisation) :

    global Vx

    x_i = sympy.Matrix([current_state_tab[1], Vx, current_state_tab[0], current_state_tab[2], current_state_tab[3], current_state_tab[4]],dtype=float_)
    u_i = sympy.Matrix(command,dtype=float_)

    state = State(current_state_tab[1], Vx, current_state_tab[0], current_state_tab[2], current_state_tab[3], current_state_tab[4], command[0], command[1])

    Bd = get_Bd(state, T_discretisation)
    Ad = get_Ad(state, T_discretisation)

    x_i = Ad*x_i + Bd*u_i

    return np.array([x_i[2],x_i[0],x_i[3],x_i[4],x_i[5]])

def get_ode_stiff_solution(current_state_tab, command, T_discretisation) :

    solver = odespy.Lsodar(deriv2, rtol=0.0, atol=1e-6, adams_or_bdf='adams', order=10, f_args=[command])
    #solver = odespy.Lsodar(deriv2, rtol=0.0, atol=1e-6, adams_or_bdf='adams', order=4, f_args=[command])
    solver.set_initial_condition(current_state_tab)
    t_points = np.linspace(0, T_discretisation, 150)
    #t_points = np.linspace(0, T_discretisation, 10)
    x, t = solver.solve(t_points)

    return x[-1]

"""
current_state_tab contient les informations relatives à l'état du robot mais sous forme de np.array
return un np.array
"""
def get_discret_following_state(current_state_tab, command, T_discretisation) :

    #return get_rk4_solution(current_state_tab, command, T_discretisation)
    #return get_matrix_solution(current_state_tab, command, T_discretisation)
    return get_ode_stiff_solution(current_state_tab, command, T_discretisation)


def constraints(minimize_variables,T_discretisation) :

    h = 0

    for i in range(6,len(minimize_variables),6) :
        state_k1    = np.array( [minimize_variables[i],
                                minimize_variables[i+1],
                                minimize_variables[i+2],
                                minimize_variables[i+3],
                                minimize_variables[i+4]]) 
        state_k0    = np.array( [minimize_variables[i-6],
                                minimize_variables[i-6+1],
                                minimize_variables[i-6+2],
                                minimize_variables[i-6+3],
                                minimize_variables[i-6+4]])
        command     = np.array([minimize_variables[i-6+5],
                                minimize_variables[i-6+5]])

        model_calc = get_discret_following_state(state_k0,command,T_discretisation)
        state_k1_calc = np.array([model_calc[0],model_calc[1],model_calc[2],model_calc[3],model_calc[4]])

        h += (abs(state_k1) - abs(state_k1_calc))**2

    #print "h : ",h

    result = 0
    for i in range(0,len(h)) :
        result += h[i]

    print "result : ",result," pour ",len(minimize_variables)

    return result

def constraints2(minimize_variables,i,T_discretisation) :

   state_k1    = np.array( [minimize_variables[i+6],
                           minimize_variables[i+6+1],
                           minimize_variables[i+6+2],
                           minimize_variables[i+6+3],
                           minimize_variables[i+6+4]]) 

   state_k0    = np.array( [minimize_variables[i],
                           minimize_variables[i+1],
                           minimize_variables[i+2],
                           minimize_variables[i+3],
                           minimize_variables[i+4]])

   command     = np.array([minimize_variables[i+5],
                           minimize_variables[i+5]])

   model_calc = get_discret_following_state(state_k0,command,T_discretisation)
   state_k1_calc = np.array([model_calc[0],model_calc[1],model_calc[2],model_calc[3],model_calc[4]])

   error = (state_k1_calc**2 - state_k1**2)**2

   result = 0
   for value in error :
        result += value

   #print result

   return result


def initiate_ref_signal(ref_starting_state, horizon_iteration) :

    global maneuvers

    # détermination de la trajectoire de référence
    print "[*] Définition d'une suite de primitives de mouvement"
    seq_fin = [maneuvers[0].copy()]

    print "[*] Définition des trajectoires de référence"
    states_ref = set_state_ref2(ref_starting_state,seq_fin)
    
    Xref = []
    Yref = []
    for i in range(0,len(states_ref)) :
        Xref.append(states_ref[i].x)
        Yref.append(states_ref[i].y)

    # définition du chemin sur une horizon
    u_ref = get_u_ref(ref_starting_state, states_ref, horizon_iteration)
    x_ref = get_x_ref(ref_starting_state, states_ref, horizon_iteration)
    t_ref = get_t_ref(ref_starting_state, states_ref, horizon_iteration)

    return u_ref, x_ref, t_ref, Xref, Yref

def display_u_from_minimize_variables(minimize_variables) :

    u = []
    for i in range(0,len(minimize_variables),6) :
        u.append(minimize_variables[i])

    plt.plot(range(0,len(u)),u)
    plt.show()

    return 0

def initial_constraint(minimize_variables,T_discretisation,initial_state, initial_command,j) :
    state_k1 = get_discret_following_state([initial_state.Vy, initial_state.Vpsi, initial_state.psi, initial_state.x, initial_state.y], [initial_command.betaf,initial_command.betar], T_discretisation)    

    return state_k1[j] - minimize_variables[j]

#following_state = []

def model_constraint(minimize_variables,T_discretisation,i,j) :

    #global following_state
    
    #global Vx

    #a = get_model_a2_matrix()
    #b = get_model_b2_matrix()

    Vy      = minimize_variables[i*6]
    Vpsi    = minimize_variables[i*6+1]
    psi     = minimize_variables[i*6+2]
    x       = minimize_variables[i*6+3]
    y       = minimize_variables[i*6+4]
    u       = minimize_variables[i*6+5]

    # trick de l'espace : on ne calcule le modèle qu'une seulle fois
    #if j == 0 :
    #    following_state = get_discret_following_state([Vy, Vpsi, psi, x, y], [u,u], T_discretisation)


    #Vy_der      = minimize_variables[6+i*6]
    #Vpsi_der    = minimize_variables[6+i*6+1]
    #psi_der     = minimize_variables[6+i*6+2]
    #x_der       = minimize_variables[6+i*6+3]
    #y_der       = minimize_variables[6+i*6+4]

    # état à l'instant i + 1
    state_k1 = get_discret_following_state([Vy, Vpsi, psi, x, y], [u,u], T_discretisation)
    #state_k1 = following_state

    print "[",i,":",j,"]diff : ",(state_k1[j] - minimize_variables[6+i*6+j])

    return state_k1[j] - minimize_variables[6+i*6+j]



def get_commands_on_horizon(T_discretisation,horizon_iteration,current_state, current_command,u_ref, x_ref, t_ref, Xref, Yref) :

    global betaMax

    print "[x_ref] len ",len(x_ref)

    # une commande = un angle de rotation = angle de braquage avant ET arrière
    #var_0 = [current_state.Vy,current_state.Vpsi,current_state.psi,current_state.x,current_state.y,current_command.betaf]*(horizon_iteration)
    var_0 = [current_state.Vy,current_state.Vpsi,current_state.psi,current_state.x,current_state.y,betaMax]*(horizon_iteration)
    #var_0 = [1,1,1,1,1,1]*(horizon_iteration)
    var_0 = [var_0]

    # etablissement des bornes
    bnds = []
    for i in range(0,len(var_0[0]),6) :
        bnds.append((-100,100))
        bnds.append((-100,100))
        bnds.append((-100,100))
        bnds.append((-100,100))
        bnds.append((-100,100))
        bnds.append((-betaMax,betaMax))

    # établissement des contraintes
    #cons = ({'type': 'eq', 'fun': constraints, 'args':[T_discretisation]})
    cons = []
    # contraintes de respect du modèle dynamique entre la prédiction et l'état actuel
    for j in range(0,5) :
        cons.append({'type': 'eq', 'fun': initial_constraint, 'args':[T_discretisation,current_state, current_command,j]})

    # contraintes de réspect du modèle dynamique sur la prédiction
    for i in range(0,horizon_iteration-1) :
        for j in range(0,5) :
            cons.append({'type': 'eq', 'fun': model_constraint, 'args':[T_discretisation,i,j]})
    #    cons.append({'type': 'eq', 'fun': constraints2, 'args':[i,T_discretisation]})
    #cons.append({'type': 'eq', 'fun': display_u_from_minimize_variables })
    #cons.append({'type': 'eq', 'fun': constraints, 'args':[T_discretisation]})
    cons = tuple(cons)

    t1 = time.time()
    # options={'maxiter': 20},
    var_optimized = minimize(fun=f_cost_MPC,x0=var_0, bounds=bnds, options={'maxiter': 10}, constraints=cons, args=[u_ref,x_ref]).x
    #u_optimized = minimize(fun=f_cost_MPC,x0=u_0, bounds=cons, options={'maxiter': 1},args=[current_state,current_command,u_ref,x_ref,t_ref,Ad,Bd]).x
    t2 = time.time()

    delta_t = t2 - t1

    print "[var_optimized] ",var_optimized

    # récupération des commandes 
    u_optimized = []
    Vy = []
    Vpsi = []
    psi = []
    x = []
    y = []
    for i in range(0,len(var_optimized),6) :
        Vy.append(var_optimized[i])
        Vpsi.append(var_optimized[i+1])
        psi.append(var_optimized[i+2])
        x.append(var_optimized[i+3])
        y.append(var_optimized[i+4])
        u_optimized.append(var_optimized[i+5])

    plt.figure("Evolution Vy")
    plt.plot(range(0,len(Vy)),Vy)
    plt.figure("Evolution Vpsi")
    plt.plot(range(0,len(Vpsi)),Vpsi)
    plt.figure("Evolution psi")
    plt.plot(range(0,len(psi)),psi)
    plt.figure("Evolution x")
    plt.plot(range(0,len(x)),x)
    plt.figure("Evolution y")
    plt.plot(range(0,len(y)),y)
    plt.figure("Evolution u")
    plt.plot(range(0,len(u_optimized)),u_optimized)
    plt.show()

    print "u optimized      : ",u_optimized
    print "computation time : ",delta_t

    # mise en forme de u optimized
    u = [[current_state.beta_f,current_state.beta_r]]
    for i in range(0,horizon_iteration-1) :
        u.append([u_optimized[i],u_optimized[i]])
    print "u : ",u

    return u, delta_t

def display_states(states) :

    Vy = []
    Vpsi = []
    psi = []
    x = []
    y = []
    u = []

    for state in states :
        Vy.append(state.Vy)
        Vpsi.append(state.Vpsi)
        psi.append(state.psi)
        x.append(state.x)
        y.append(state.y)
        u.append(state.beta_f)

    plt.figure("Evolution Vy")
    plt.plot(range(0,len(Vy)),Vy)
    plt.figure("Evolution Vpsi")
    plt.plot(range(0,len(Vpsi)),Vpsi)
    plt.figure("Evolution psi")
    plt.plot(range(0,len(psi)),psi)
    plt.figure("Evolution x")
    plt.plot(range(0,len(x)),x)
    plt.figure("Evolution y")
    plt.plot(range(0,len(y)),y)
    plt.figure("Evolution u")
    plt.plot(range(0,len(u)),u)
    plt.show()

def main() :
    
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

    # initialisation des signaux de référence
    #u_ref, x_ref, t_ref, Xref, Yref = initiate_ref_signal(State(0,4,0,0,0,0,0,0,0))

    
    ####### DEBUT DES TESTS D'INCIDENCE DES PARAMETRES ########


    # valeurs à tester
    T_discretisation_sets   = [0.1,0.2,0.3,0.4]
    horizon_iteration_set   = [5,10,15,20]
    
    #T_discretisation_sets   = [0.5]
    #horizon_iteration_set   = [30]
    times_results           = []

    for T in T_discretisation_sets :

        T_discretisation = T

        X_results               = []
        Y_results               = []
        

        for horizon in horizon_iteration_set :

            horizon_iteration = horizon

            print " ## "
            print " ##  Calcul de (T_discretisation : ",str(T_discretisation)," ; ",str(horizon_iteration),")"
            print " ## "
            
            # etats intial
            perturbation_y  = 0.1
            orientation     = np.radians(0)
            current_state   = State(0,4,0,orientation,0,perturbation_y,0,0,0)
            current_command = Command(0,0)

            # chargement de la trajectoire de référence
            ##
            ## C'est ici que l'intégration des primitives de mouvement va se faire
            ##
            set_maneuver_0(horizon_iteration,T_discretisation)
            u_ref, x_ref, t_ref, Xref, Yref = initiate_ref_signal(State(0,4,0,0,0,0,0,0,0), horizon_iteration)
            
            # génération de la commande
            u, duration = get_commands_on_horizon(T_discretisation,horizon_iteration,current_state,current_command,u_ref, x_ref, t_ref, Xref, Yref) 
                       
            times_results.append(duration)
            
            # simulation de la rectification sur la trajectoire de référence
            X, Y, states = simulate(u, current_state, T_discretisation)

            X_results.append(X)
            Y_results.append(Y)
            

        # affichage des résultats
        plt.figure("Variation de X pour différentes horizons avec un pas de "+str(T_discretisation)+"s")
        plt.plot(range(0,len(Xref)), ':', Xref, label='$Ref$')
        i_horizon = 0
        for X in X_results :
            plt.plot(range(0,len(X)), X, label='$ T_{horizon} = '+str(horizon_iteration_set[i_horizon])+'$')
            i_horizon += 1
        plt.legend()
        plt.savefig('X-'+str(T_discretisation)+'.jpg')

        plt.figure("Variation de Y pour différentes horizons avec un pas de "+str(T_discretisation)+"s")
        plt.plot(range(0,len(Yref)), ':', Yref, label='$Ref$')
        i_horizon = 0
        for Y in Y_results :
            plt.plot(range(0,len(Y)), Y, label='$ T_{horizon} = '+str(horizon_iteration_set[i_horizon])+'$')
            i_horizon += 1
        plt.legend()
        plt.savefig('Y-'+str(T_discretisation)+'.jpg')

        # calcul des erreurs
        # calcul de l'erreur quadratique selon y
        
        i_horizon = 0
        for Y in Y_results :
            delta_y = []
            for i in range(0,min(len(Yref),len(Y))) :
                delta_y.append((Yref[i]-Y[i])**2)
            # calcul de l'erreur quadratique moyenne selon y
            eqm = np.array(delta_y).mean()
            eqm_array_y = []
            for i in range(0,len(Yref)) :
                eqm_array_y.append(eqm)

            plt.figure("Erreur quadratique selon y (pas : "+str(T_discretisation)+" - horizon : "+str(horizon_iteration_set[i_horizon])+")")
            plt.plot(range(0,len(delta_y)), delta_y)
            plt.plot(range(0,len(eqm_array_y)), eqm_array_y)
            plt.savefig('ErreurQuad-Y-'+str(T_discretisation)+'-'+str(horizon_iteration_set[i_horizon])+'.jpg')
            i_horizon += 1

        # calcul de l'erreur quadratique selon x
        i_horizon = 0
        for X in X_results :
            delta_x = []
            for i in range(0,min(len(Xref),len(X))) :
                delta_x.append((Xref[i]-X[i])**2)
            # calcul de l'erreur quadratique moyenne selon x
            eqm = np.array(delta_x).mean()
            eqm_array_x = []
            for i in range(0,len(Xref)) :
                eqm_array_x.append(eqm)

            plt.figure("Erreur quadratique selon x (pas : "+str(T_discretisation)+" - horizon : "+str(horizon_iteration_set[i_horizon])+")")
            plt.plot(range(0,len(delta_x)), delta_x)
            plt.plot(range(0,len(eqm_array_x)), eqm_array_x)
            plt.savefig('ErreurQuad-X-'+str(T_discretisation)+'.jpg')
            i_horizon += 1


    # info sur les temps d'executions
    print " ## Temps de calcul"
    mon_fichier = open("time_result.csv", "w")  
    mon_fichier.write("Pas de discretisation [s],Horizon [nb de pas de discretisation],Temps de calcul du probleme de minimisation [s]\n")
    
    print times_results
    i_time = 0
    for T in T_discretisation_sets :
        for horizon in horizon_iteration_set :
            print " (T_discretisation : ",str(T)," ; ",str(horizon),") : ",str(times_results[i_time]),' s'
            mon_fichier.write(str(T)+','+str(horizon)+','+str(times_results[i_time])+'\n')
            i_time += 1
    mon_fichier.close()

    # calcul de l'erreur quadratique selon y
    # delta_y = []
    # for i in range(0,min(len(Yref),len(Y))) :
    #     delta_y.append((Yref[i]-Y[i])**2)
    # print "quad error y simu :",delta_y

    # # calcul de l'erreur quadratique moyenne selon y
    # eqm = np.array(delta_y).mean()
    # eqm_array_y = []
    # for i in range(0,len(Yref)) :
    #     eqm_array_y.append(eqm)

    # # calcul de l'erreur quadratique selon x
    # delta_x = []
    # for i in range(0,min(len(Xref),len(X))) :
    #     delta_x.append((Xref[i]-X[i])**2)
    # print "quad error x simu :",delta_x

    # # calcul de l'erreur quadratique moyenne selon x
    # eqm = np.array(delta_x).mean()
    # eqm_array_x = []
    # for i in range(0,len(Xref)) :
    #     eqm_array_x.append(eqm)


    # # affichage des trajectoires obtenues
    # plt.figure("Evolution de la trajectoire du robot")
    # plt.plot(Xref, Yref)
    # plt.plot(X,Y)
    # plt.show()

    # # affichage des variations sur x
    # plt.figure("Evolution de la trajectoire du robot selon x")
    # plt.plot(range(0,len(Xref)), Xref)
    # plt.plot(range(0,len(X)), X)
    # plt.show()

    # # affichage des variations sur y
    # plt.figure("Evolution de la trajectoire du robot selon y")
    # plt.plot(range(0,len(Yref)), Yref)
    # plt.plot(range(0,len(Y)), Y)
    # plt.show()

    # # affichage des erreurs
    # plt.figure("Erreur quadratique selon x")
    # plt.plot(range(0,len(delta_x)), delta_x)
    # plt.plot(range(0,len(eqm_array_x)), eqm_array_x)
    # plt.show()

    # plt.figure("Erreur quadratique selon y")
    # plt.plot(range(0,len(delta_y)), delta_y)
    # plt.plot(range(0,len(eqm_array_y)), eqm_array_y)
    plt.show()

def test_model() :

    T_discretisation = 0.3
    horizon_iteration = 50

    u = []
    for i in range(0,int(horizon_iteration/2)) :
        u.append([0.69,0.69])
    for i in range(int(horizon_iteration/2),horizon_iteration) :
        u.append([-0.1,-0.1])

    t1 = time.time()
    X,Y,states = simulate(u,State(0,4,0,0,0,0,0,0,0),T_discretisation)
    t2 = time.time()

    print "Execution time : ",(t2-t1)

    #display_states(states)
    plt.plot(X,Y)
    plt.show()

if __name__ == '__main__':
    main()
    #a = get_model_a_matrix()
    #b = get_model_b_matrix()

    #print "a : ",a
    #{print "b : ",b
    #test_model()