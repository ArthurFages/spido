#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import odespy

from parameters import *
from model import *


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

def get_explicit_euler_discret_following_state(current_state_tab, command, T_discretisation) :
	
	global Vx

	Vpsi	= current_state_tab[1]
	Vy		= current_state_tab[0] 
	psi		= current_state_tab[2] 
	x		= current_state_tab[3]
	y		= current_state_tab[4] 
	beta_f 	= command[0] 
	beta_r 	= command[1]

	A = get_model_a3_matrix_lin(State(Vpsi, Vx, Vy, psi, x, y, beta_f, beta_r))
	b = get_model_b2_matrix()
	B = np.array([[b[0],b[1]],[b[4],b[5]],[0,0],[0,0],[0,0]])

	#print "A.dot(current_state_tab) : ",A.dot(current_state_tab)
	#print "B : ",B.dot(command)
	#print "command : ",command

	return (A.dot(current_state_tab) + B.dot(command))*T_discretisation + current_state_tab

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

    #print "initial state ; ",stable_current_state.to_string_v()
    #print "T discretisation ; ",T_discretisation
    #print "horizon : ",horizon_iteration

    # commande et états simulés à l'aide du modèle
    x_i     = np.array([stable_current_state.Vy, stable_current_state.Vpsi, stable_current_state.psi, stable_current_state.x, stable_current_state.y])
    u_i     = np.array(u[0])
    
    states = [current_state]

    #print "states[-1] : ",states[-1]

    #T = T_discretisation
    t = T_discretisation

    for i in range(0,len(u)):

        #print "Commande : ",u_i

        u_i = u[i]

        #print states[-1].to_string_v()

        x_i = get_discret_following_state(x_i, u_i, T_discretisation)
        #x_i = get_ode_stiff_solution(x_i, u_i, T_discretisation)

        # enregistement de l'état actuel
        states.append(State(x_i[1],Vx,x_i[0],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1],t))

        #u_i = u[i]
        
        # chargement de la commande prédite à l'itération suivante, et de la commande et trajectoire de référence à l'itération suivante
        X.append(x_i[3])
        Y.append(x_i[4])

        t += T_discretisation

        #print "Result : (",x_i[3],"",x_i[4],")"

        #print "Nouvel etat :",x_i

    #print "Pas de discretisation : ",T_discretisation
                                

    return X, Y, states
