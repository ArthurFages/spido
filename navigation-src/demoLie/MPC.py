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
import mpmath
import eq_diff as eq
import scipy.linalg as sclin


from sympy.abc import x
from scipy.integrate import quad
from scipy.optimize import basinhopping, minimize, fmin_cg
from scipy.linalg import expm3, expm2, expm
from pyOpt import *


from math import fabs
from scipy.integrate import quad
from geometry_msgs.msg import Pose2D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from random import randint
from mpl_toolkits.mplot3d import Axes3D


#import tf
#from dynamic_reconfigure.server import Server
#from spido_navigation.cfg import DriverConfig
from std_srvs.srv import Empty

# import usefull classes
from parameters import *
from classes import *
from model import *

# reccursion setup
sys.setrecursionlimit(reccursion_limit)

# ========================== UTILITY FUNCTIONS =============================================
    
def fast_integration(f,t0,tf,dt) :
    x = sympy.Symbol("x")
    result  = 0
    t       = t0
    
    while t<tf :
        result = result + abs(f.evalf(subs={x: t})*dt)
        t = t + dt
    
    return result

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
def integrate_m(M,len_lin,len_col,t0,tf,dt) :
    x = sympy.Symbol("x")
    result = sympy.zeros(len_lin,len_col)
    for i in range(0,len_lin) :
        for j in range(0,len_col) :
            result[i,j] = fast_integration(M[i,j],t0,tf,dt)
    return result


# ============================================================================================

# ========================== DISPLAY SIGNAL ==================================================
def display_state_ref_simu() :
    
    global states_ref
    global simu_states

    global erreurs_u

    # mise à l'échelle
    size_ref    = len(states_ref)
    size_simu   = len(simu_states)

    size_graph = size_simu
    if size_ref<size_simu :
        size_graph = size_ref
    
    # trajectoires de référence
    Vx_ref      = []
    Vy_ref      = []
    Vpsi_ref    = []
    psi_ref     = []
    x_ref       = []
    y_ref       = []
    t_ref       = []
    b_f_ref     = []
    b_r_ref     = []
    i_ref       = 0
    
    for state in states_ref :
        if i_ref >= size_graph :
            break
        Vx_ref.append(state.Vx)
        Vy_ref.append(state.Vy)
        Vpsi_ref.append(state.Vpsi)
        x_ref.append(state.x)
        y_ref.append(state.y)
        psi_ref.append(state.psi)
        t_ref.append(state.t)
        b_f_ref.append(state.beta_f)
        b_r_ref.append(state.beta_r)
        i_ref += 1
        
    # trajectoires relevées
    Vx_sim      = []
    Vy_sim      = []
    Vpsi_sim    = []
    psi_sim     = []
    x_sim       = []
    y_sim       = []
    t_sim       = []
    b_f_sim     = []
    b_r_sim     = []
    i_simu      = 0
    
    for state in simu_states :
        if i_simu >= size_graph :
            break
        Vx_sim.append(state.Vx)
        Vy_sim.append(state.Vy)
        Vpsi_sim.append(state.Vpsi)
        x_sim.append(state.x)
        y_sim.append(state.y)
        psi_sim.append(state.psi)
        t_sim.append(state.t)
        b_f_sim.append(state.beta_f)
        b_r_sim.append(state.beta_r)
        i_simu += 1

    # calcul des erreurs de trajectoire
    erreurs_x = []
    erreurs_y = []

    for i in range(0,len(x_sim)) :
        erreurs_x.append(x_ref[i]-x_sim[i])
    for i in range(0,len(y_sim)) :
        erreurs_y.append(y_ref[i]-y_sim[i])
    

    
    plt.figure("Evolution etat référentiel du SPIDO")
    plt.subplot(2,2,1)
    plt.title("x au cours du temps")
    plt.plot(range(0,len(x_ref)),x_ref)
    plt.plot(range(0,len(x_sim)),x_sim)
    plt.legend(["ref","simu"])
    plt.subplot(2,2,2)
    plt.title("y au cours du temps")
    plt.plot(range(0,len(y_ref)),y_ref)
    plt.plot(range(0,len(y_sim)),y_sim)
    plt.legend(["ref","simu"])
    plt.subplot(2,2,3)
    plt.title("Psi au cours du temps")
    plt.plot(range(0,len(psi_ref)),psi_ref)
    plt.plot(range(0,len(psi_sim)),psi_sim)
    plt.legend(["ref","simu"])
    plt.subplot(2,2,4)
    plt.title("Vpsi au cours du temps")
    plt.plot(range(0,len(Vpsi_ref)),Vpsi_ref)
    plt.plot(range(0,len(Vpsi_sim)),Vpsi_sim)
    plt.legend(["ref","simu"])
    
    
    plt.figure("Evolution etat référentiel du SPIDO - 2")
    plt.subplot(2,2,1)
    plt.title("Vx au cours du temps")
    plt.plot(range(0,len(Vx_ref)),Vx_ref)
    plt.plot(range(0,len(Vx_sim)),Vx_sim)
    plt.legend(["ref","simu"])
    plt.subplot(2,2,2)
    plt.title("Vy au cours du temps")
    plt.plot(range(0,len(Vy_ref)),Vy_ref)
    plt.plot(range(0,len(Vy_sim)),Vy_sim)
    plt.legend(["ref","simu"])
    plt.subplot(2,2,3)
    plt.title("Beta r au cours du temps")
    plt.plot(range(0,len(b_r_ref)),b_r_ref)
    plt.plot(range(0,len(b_r_sim)),b_r_sim)
    plt.legend(["ref","simu"])
    plt.subplot(2,2,4)
    plt.title("Beta f au cours du temps")
    plt.plot(range(0,len(b_f_ref)),b_f_ref)
    plt.plot(range(0,len(b_f_sim)),b_f_sim)
    plt.legend(["ref","simu"])

    plt.figure("Trajectoire du SPIDO sur le plan")
    plt.plot(x_ref,y_ref)
    plt.plot(x_sim,y_sim)
    plt.legend(["ref","simu"])
    plt.show()

    # affichage des erreurs
    plt.figure("Ecart entre la commande de référence et la commande calculée")
    plt.plot(range(0,len(erreurs_u)),erreurs_u)
    plt.show()

    plt.figure("Erreur sur x")
    plt.plot(range(0,len(erreurs_x)),erreurs_x)
    plt.show()

    plt.figure("Erreur sur y")
    plt.plot(range(0,len(erreurs_y)),erreurs_y)
    plt.show()

# ============================================================================================

# ========================== REFERENCE SIGNAL ================================================

def get_t_ref(current_position, states_ref, horizon_iteration) :
    
    global STOP_PROCESS
    
    #global horizon_iteration
    #global states_ref
    
    stable_current_position = current_position.copy()
    
    # identification de la primitive dans laquelle on se situe
    i_t = 0
    delta = 1e122
    for i in range(0, len(states_ref)) :
        delta_i = np.sqrt((states_ref[i].x - stable_current_position.x)**2 + (states_ref[i].y - stable_current_position.y)**2)
        if  delta_i < delta :
            delta = delta_i
            i_t = i
            
    # prise en compte de la fin de la trajectoire
    STOP_PROCESS = (i_t==len(states_ref)-1)
    
    result = []
    if len(states_ref)-i_t < horizon_iteration :
        for i in range(i_t,len(states_ref)) :
            result.append(states_ref[i].t)
    else :
        for i in range(i_t,i_t+horizon_iteration) :
            result.append(states_ref[i].t)
        
    return result

def get_t_ref_extended(current_position, states_ref, horizon_iteration) :
    
    global STOP_PROCESS
    
    #global horizon_iteration
    #global states_ref
    
    stable_current_position = current_position.copy()
    
    # identification de la primitive dans laquelle on se situe
    i_t = 0
    delta = 1e122
    for i in range(0, len(states_ref)) :
        delta_i = np.sqrt((states_ref[i].x - stable_current_position.x)**2 + (states_ref[i].y - stable_current_position.y)**2)
        if  delta_i < delta :
            delta = delta_i
            i_t = i
            
    # prise en compte de la fin de la trajectoire
    STOP_PROCESS = (i_t==len(states_ref)-1)

    i0 = i_t
    if i0 > 0 :
        i0 = i0 - 1
    
    result = []
    if len(states_ref)-i0 < horizon_iteration :
        for i in range(i_0,len(states_ref)) :
            result.append(states_ref[i].t)
    else :
        for i in range(i0,i0+horizon_iteration) :
            result.append(states_ref[i].t)

    return result

def get_x_ref(current_position, states_ref, horizon_iteration) :
    
    
    global STOP_PROCESS
    
    #global horizon_iteration
    #global states_ref
    
    stable_current_position = current_position.copy()

    print "[horizon_iteration] ",horizon_iteration
    
    # identification de la primitive dans laquelle on se situe
    i_x = 0
    delta = 1e122
    for i in range(0, len(states_ref)) :
        delta_i = np.sqrt((states_ref[i].x - stable_current_position.x)**2 + (states_ref[i].y - stable_current_position.y)**2)
        if  delta_i < delta :
            delta = delta_i
            i_x = i
            
    # prise en compte de la fin de la trajectoire
    STOP_PROCESS = (i_x==len(states_ref)-1)
    
    result = []
    if len(states_ref)-i_x < horizon_iteration :
        for i in range(i_x,len(states_ref)) :
            #result.append(np.array([states_ref[i].Vy,states_ref[i].Vpsi,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
            result.append(np.array([states_ref[i].Vpsi,states_ref[i].Vx,states_ref[i].Vy,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
    else :
        for i in range(i_x,i_x+horizon_iteration) :
            #result.append(np.array([states_ref[i].Vy,states_ref[i].Vpsi,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
            result.append(np.array([states_ref[i].Vpsi,states_ref[i].Vx,states_ref[i].Vy,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))

    return result

def get_x_ref_extended(current_position, states_ref, horizon_iteration) :
    
    global STOP_PROCESS
    
    #global horizon_iteration
    #global states_ref
    
    stable_current_position = current_position.copy()

    print "[horizon_iteration] ",horizon_iteration
    
    # identification de la primitive dans laquelle on se situe
    i_x = 0
    delta = 1e122
    for i in range(0, len(states_ref)) :
        delta_i = np.sqrt((states_ref[i].x - stable_current_position.x)**2 + (states_ref[i].y - stable_current_position.y)**2)
        if  delta_i < delta :
            delta = delta_i
            i_x = i
            
    # prise en compte de la fin de la trajectoire
    STOP_PROCESS = (i_x==len(states_ref)-1)

    i0 = i_x
    if i0 > 0 :
        i0 = i0 - 1
    
    result = []
    if len(states_ref)-i0 < horizon_iteration :
        for i in range(i0,len(states_ref)) :
            #result.append(np.array([states_ref[i].Vy,states_ref[i].Vpsi,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
            result.append(np.array([states_ref[i].Vpsi,states_ref[i].Vx,states_ref[i].Vy,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
    else :
        for i in range(i0,i0+horizon_iteration) :
            #result.append(np.array([states_ref[i].Vy,states_ref[i].Vpsi,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
            result.append(np.array([states_ref[i].Vpsi,states_ref[i].Vx,states_ref[i].Vy,states_ref[i].psi,states_ref[i].x,states_ref[i].y],dtype=float_))
        
    return result
    
def get_u_ref(current_position, states_ref, horizon_iteration) :
    
    global STOP_PROCESS
    
    #global horizon_iteration
    #global states_ref
    
    stable_current_position = current_position.copy()
    
    # identification de la primitive dans laquelle on se situe
    i_u = 0
    delta = 1e122
    for i in range(0, len(states_ref)) :
        delta_i = np.sqrt((states_ref[i].x - stable_current_position.x)**2 + (states_ref[i].y - stable_current_position.y)**2)
        if  delta_i < delta :
            delta = delta_i
            i_u = i
            
    # prise en compte de la fin de la trajectoire
    STOP_PROCESS = (i_u==len(states_ref)-1)
    
    print "get u ref : ",i_u
    print "delta : ",delta
    #print "states ref ",states_ref[i_u].to_string_v()
    #print "curr state ",stable_current_position.to_string_v()
    
    result = []
    if len(states_ref)-i_u < horizon_iteration :
        for i in range(i_u,len(states_ref)) :
            result.append(np.array([states_ref[i].beta_f,states_ref[i].beta_r],dtype=float_))
    else :
        for i in range(i_u,i_u+horizon_iteration) :
            result.append(np.array([states_ref[i].beta_f,states_ref[i].beta_r],dtype=float_))
        
    return result

"""
Donne une série de 3 state de x_ref et t_ref qui se suivent autour de current_state
"""
def get_xt_ref_k(current_state, T_discretisation, x_ref, t_ref) :

    #print "xref 0 : ",x_ref[-1]

    nul_array = np.array([0,0,0,0,0,0])

    # on récupère l'indice du state de x_ref qui est le plus proche de current state
    i_x = 0
    delta = 1e122
    for i in range(0, len(x_ref)) :
        delta_i = np.sqrt((x_ref[i][4] - current_state.x)**2 + (x_ref[i][5] - current_state.y)**2)
        if  delta_i < delta :
            delta = delta_i
            i_x = i

    # on prend l'état suivant et l'état précédent
    x_ref_k = []
    t_ref_k = []

    if i_x == 0 :

        # cas initial
        result = x_ref[i_x:i_x+2]
        result.insert(0,nul_array)
        x_ref_k = result

        result = t_ref[i_x:i_x+2]
        result.insert(0,-T_discretisation)
        t_ref_k = result

    elif i_x == len(x_ref)-1 :

        # cas final
        result = x_ref[i_x-1:i_x+1]
        result.append(x_ref[i_x])
        x_ref_k = result

        result = t_ref[i_x-1:i_x+1]
        result.append(t_ref[-1]+T_discretisation)
        t_ref_k = result
    else :
        
        x_ref_k = x_ref[i_x-1:i_x+2]
        t_ref_k = t_ref[i_x-1:i_x+2]

    # on retourne le tout à la fonction appellante avec allégresse
    return x_ref_k, t_ref_k


# ============================================================================================

current_state   = None
current_command = None

#b = get_model_b_matrix()

# récupération des paramètres
u_ref = None
x_ref = None
t_ref = None

ind = 0

STOP_PROCESS = False
etats = []
Ad_mat = None
A_mat = None
Bd_mat = None
B_mat = None
T_mat = None



"""
Donne le modèle discrétisé du robot à un état donné en fonction d'un pas de discrétisation
"""

def get_discretized_model(current_state,T) :

    global exponential_factor

    A   = get_model_a2_matrix_lin(current_state.copy())
    b   = get_model_b_matrix()
    B   = np.matrix([[b[4],b[5]],[0,0],[b[0],b[1]],[0,0],[0,0],[0,0]])
            
    Ad  = expm(A*T)

    #As  = sympy.Matrix(A)
    #Bs  = sympy.Matrix(B)
    x   = sympy.Symbol("x")

    t1 = time.time()
    exp_at = fast_expm(sympy.Matrix(A)*x,6,exponential_factor)
    t2 = time.time()

    print "Temps de calcul de l'exponentielle : ",(t2-t1)

    to_integrate = exp_at*B

    #print "expt at : ",exp_at
    #print "to integrate : ",to_integrate
    
    t1 = time.time()
    Bd = integrate_m(to_integrate,6,2,0,T,T/integration_factor)
    t2 = time.time()

    print "Temps de calcul de l'integration : ",(t2-t1)

    return Ad, Bd

"""
Fonction de cout utilisée pour a génération de commande par MPC
x : [horizon,u0,u1,...uhorizon]
Hyp : on part du principe que pour une commande, l'angle de braquage avant = angle de braquage arrière
"""
def f_cost_MPC(minimize_variables,args):#current_state,current_command) :

    u_ref = args[0]
    x_ref = args[1]

    #print "x_ref : ",x_ref
    
    # initialisation
    f_cost  = 0
    
    #==========================================================================
    K = [1,1,1,1,100]
    Q = 1
    R = 1
    S = 1
    #==========================================================================

    u_i_prev = np.array([])

    i_ref = 0
    print "--------------------------------------------------------------------------------------"
    for i in range(0,len(minimize_variables),6):

        x_i = np.array( [minimize_variables[i],
                        minimize_variables[i+1],
                        minimize_variables[i+2],
                        minimize_variables[i+3],
                        minimize_variables[i+4]])

        #x_i = np.array( [minimize_variables[i+3],
        #                minimize_variables[i+4]])

        u_i = np.array( [minimize_variables[i+5],
                        minimize_variables[i+5]])

        x_ref_i = np.array([x_ref[i_ref][2],
                            x_ref[i_ref][0],
                            x_ref[i_ref][3],
                            x_ref[i_ref][4],
                            x_ref[i_ref][5]])

        #x_ref_i = np.array([x_ref[i_ref][4],
        #                    x_ref[i_ref][5]])

        u_ref_i = u_ref[i_ref]

        #print " u ",str(u_i)
        #print " xi   ",str(x_i)
        #print "uref - u = ",u_ref_i-u_i
        print "xref - x = ",x_ref_i-x_i


        if len(u_i_prev) == 0 :
            # cas du premier etat = il n'existe pas d'état précédents à calculer
            f_cost = 0.5*(Q*(K*(x_ref_i-x_i)).dot(K*(x_ref_i-x_i)) + R*(u_ref_i-u_i).dot(u_ref_i-u_i))
            #print "first f_cost : ",f_cost
        else :
            #print "Q*(x_ref_i-x_i).dot(x_ref_i-x_i) = ",Q*(x_ref_i-x_i).dot(x_ref_i-x_i)
            #print "R*(u_ref_i-u_i).dot(u_ref_i-u_i) = ",R*(u_ref_i-u_i).dot(u_ref_i-u_i) 
            #print "S*(u_i-u_i_prev).dot(u_i-u_i_prev) = ",S*(u_i-u_i_prev).dot(u_i-u_i_prev)
            f_cost = f_cost + 0.5*(Q*(K*(x_ref_i-x_i)).dot(K*(x_ref_i-x_i)) + R*(u_ref_i-u_i).dot(u_ref_i-u_i) + S*(u_i-u_i_prev).dot(u_i-u_i_prev))
            #print "ui+1 - ui =",u_i-u_i_prev
            #print "other fcost : ",f_cost
        u_i_prev = u_i

        i_ref += 1

    print "fcost : ",f_cost
    return f_cost

def mybounds(**kwargs):
    global alpha_max
    x = kwargs["x_new"]
    amax = bool(np.all(x <= alpha_max))
    amin = bool(np.all(x >= -alpha_max))
    return amax and amin

# == Génération de la commande ===================================================================================

"""
Return the actual command to the user, depending on the sequence of motion primitives and the current position
We'll use Model Predictive Command
"""
def get_command_motion_seq(current_state_arg,current_command_arg) :
    
    global ind
    global etats
    
    global erreurs_u
    
    global horizon_iteration # nombre de commandes prédites
    global states_ref
    global current_command
    global current_state
    
    global u_ref
    global x_ref
    global t_ref
    
    global Ad_mat
    global A_mat
    global Bd_mat
    global B_mat
    global T_mat
    
    # partage des ressources de position
    current_state   = current_state_arg
    current_command = current_command_arg
    
    u_ref = get_u_ref(current_state)
    x_ref = get_x_ref(current_state)
    t_ref = get_t_ref(current_state)
    
    #print "t_ref : ",t_ref
    #print "u_ref : ",u_ref
        
    # détermination des prévisions de commande initial
    u_0 = []
    for i in range(0,horizon_iteration) :
        u_0 += [0,0]#[current_command.betaf, current_command.betar]
    
    u_0 = [u_0]
     
    t1 = time.time()
    #u_optimized = basinhopping(func=f_cost_MPC, x0=u_0, accept_test=mybounds, niter=1).x
    u_optimized = minimize(fun=f_cost_MPC, x0=u_0).x
    t2 = time.time()

    # calcul des erreurs
    erreurs_u.append(u_optimized[0]-u_ref[0])
    
    #print "nombre de pouillemes : ",ind
    print "u optimized      : ",u_optimized
    print "computation time : ",(t2-t1)
    #print "erreur : ",erreur
    
    
    # recalcul pour illustration des états pour la fonction de coûts avec u_opitmized
    #f_cost_MPC(u_optimized)
    #print "etat suivant : "
    #for etat in etats :
    #    print etat
        
    #print "Ad et Bd : "
    #print Ad_mat
    #print Bd_mat
    #print "A et B ; "
    #print A_mat
    #print B_mat
    #print "T : ",T_mat
    
    return Command(u_optimized[0],u_optimized[1])

def get_commands_on_horizon_Lie(T_discretisation,horizon_iteration,current_state, current_command,u_ref, x_ref, t_ref, Xref, Yref) :
    
    t1 = time.time()
    D = get_D_matrix(current_state)
    K = get_K_matrix(horizon_iteration, T_discretisation)
    E = get_E_matrix(current_state, x_ref, t_ref, T_discretisation)

    #print "inv : ",inv

    u = -sclin.pinv(np.transpose(D).dot(D)).dot(np.transpose(D)).dot(K).dot(E)
    t2 = time.time()

    return [u], (t2-t1) 

# ==============================================================================================================

    
"""
Transforme une séquence de primitives de mouvement en succession d'état
"""
def set_state_ref2(initial_state,seq_fin) :
    
    
    #global states_ref
    global maneuvers
    #global seq_fin
    new_state = initial_state 
    
    states_ref = []
    
    print "seq fin : ",seq_fin
    
    for i in range(0,len(seq_fin)) :

        print "# maneuver ",seq_fin[i].id_maneuver
        
        t       = float(new_state.t)
        #print "t init : ",t
        x       = float(new_state.x)
        y       = float(new_state.y)
        psi     = float(new_state.psi)
        Vpsi    = float(new_state.Vpsi)
        Vx      = float(new_state.Vx)
        Vy      = float(new_state.Vy)

        #print " State : ",[Vpsi,Vx,Vy,psi,x,y,t]
        
        #
        # /!\ les variations de x et y sont donnés dans le repère global
        #
        
        psi_0 = psi
        
        #state_trim = []
        man_states = []

        for j in range(0,len(seq_fin[i].states)-1) :
            man_states.append(seq_fin[i].states[j+1].copy())
            man_states[j].t     = t + float(man_states[j].t)
            #print "t : ",man_states[j].t
            man_states[j].psi   = psi + float(man_states[j].psi)
            
            man_states_x = man_states[j].x
            man_states_y = man_states[j].y
            
            man_states[j].x     = x + float(man_states_x)*np.cos(psi_0) - float(man_states_y)*np.sin(psi_0)
            man_states[j].y     = y + float(man_states_y)*np.cos(psi_0) + float(man_states_x)*np.sin(psi_0)

            #print "(",man_states[j].x,";",man_states[j].y,") at t=",man_states[j].t

            man_states[j].Vpsi  = float(man_states[j].Vpsi)
            man_states[j].Vx    = float(man_states[j].Vx)
            man_states[j].Vy    = float(man_states[j].Vy)   
            
        states_ref = states_ref + man_states
        
        new_state = states_ref[-1]
        #print "new state = ",new_state.to_string_v()
        #new_state = new_state_from_maneuver(new_state,seq_fin[i].q_to,seq_fin[i].id_maneuver)
        new_state.t = float(states_ref[-1].t) #+ (float(states_ref[-1].t) - float(states_ref[-1].t))
        
        #print "> New State : ",new_state.to_string()


    return states_ref
    

"""
Mise en oeuvre de la commande par MPC
Lien avec Gazebo
"""
def test_mpc() :
    
    global STOP_PROCESS
    
    global seq_fin
    
    global X
    global PSI
    global Y
    global VX
    global VY
    global VPSI
    global t
    
    global t_0
    
    global simu_states
    global Vx
    
    #VX = Vx # Vx supposée constante
    
    print "[*] Définition d'une suite de primitives de mouvement"
    
    seq_fin = [maneuvers[0].copy()]
    #seq_fin[0].thau = 3 # lancement du véhicule
    
    print "[*] Définition des trajectoires de référence"
    
    set_state_ref2(State(0,4,0,0,0,0,0,0,0))
    #print "> Affichage de ces trajectoires"
    #display_state_ref()
    
    print "[*] Suivi de ces trajectoires par MPC sous gazebo"
    
    # position initiale
    current_state   = State(0,4,0,0,0,0,0,0,0)
    current_command = Command(0,0)
    
    rospy.init_node('trajectory_following_MPC')
    
    # expression de la commande 
    cmd_safe_publisher  = rospy.Publisher('cmd_car_safe',cmd_car,queue_size=10)
    cmd_publisher       = rospy.Publisher('cmd_car',cmd_car,queue_size=10)
    state_publisher     = rospy.Publisher('state',Int32,queue_size=10)

    # simulation en indiquant directement l'angle de commande sur l'objet 
    # gazebo steering
    front_r_steer_pub   = rospy.Publisher('spido/front_right_steering/command',Float64,queue_size=10)
    front_l_steer_pub   = rospy.Publisher('spido/front_left_steering/command',Float64,queue_size=10)
    rear_r_steer_pub    = rospy.Publisher('spido/rear_right_steering/command',Float64,queue_size=10)
    rear_l_steer_pub    = rospy.Publisher('spido/rear_left_steering/command',Float64,queue_size=10)

    # listeners
    rospy.Subscriber("odom", Odometry, new_position)
    rospy.Subscriber("clock", Clock, new_time)
    
    # Command
    cmd = cmd_car()

    # taux déchatillonnage avec ROS
    r = rospy.Rate(100) # alleeeeeez, 100hz dans ta faaaace
    
    t_0 = t # début de la simulation
    
    simu_states = []
    
    # commande de la vitesse du SPIDO
    
    cmd.linear_speed = Vx
    cmd.steering_angle = current_command.betaf
    time.sleep(1)
    cmd_publisher.publish(cmd)
    cmd_safe_publisher.publish(cmd)
    state_publisher.publish(1)
    time.sleep(1)

    rospy.wait_for_service('gazebo/pause_physics')
    pause_physics = rospy.ServiceProxy("gazebo/pause_physics", Empty)

    rospy.wait_for_service('gazebo/unpause_physics')
    unpause_physics = rospy.ServiceProxy("gazebo/unpause_physics", Empty)
    
    for i in range(0,3000) :
        
        if STOP_PROCESS :
            break
            
        print "== Commandes ",i
        
        # pause de la simulation sous Gazebo
        print "[!] Pause physique"
        pause_physics()
        #rospy.wait_for_service('gazebo/pause_physics')
        #rospy.ServiceProxy("gazebo/pause_physics", Empty)

        # calcul commande
        current_command = get_command_motion_seq(current_state, current_command)
        
        # reprise de la simulation sous Gazebo
        unpause_physics()
        #rospy.wait_for_service('gazebo/unpause_physics')
        #rospy.ServiceProxy("gazebo/unpause_physics", Empty)
        print "[!] Fin de la pause physique"

        # commande directe des angles de braquage
        front_r_steer_pub.publish(Float64(current_command.betaf))
        front_l_steer_pub.publish(Float64(current_command.betaf))
        rear_r_steer_pub.publish(Float64(current_command.betar))
        rear_l_steer_pub.publish(Float64(current_command.betar))
        print "[!] Fin commande des angles"
        
        # relevé position
        current_state = State(VPSI, VX, VY, PSI, X, Y, current_command.betaf, current_command.betar, t) 
        print "[!] Fin relevé des positions"

        # enregistrement de la trajectoire
        simu_states.append(current_state)
        print "[!] Fin enregistrement des trajectoires"
        
        r.sleep()
        print "[!] Fin rospy sleep"
    
    print "> Comparaison des trajectoires planifiées et réelles"
    display_state_ref_simu()
    

def test_minimum_f_cost() :
    
    global erreur
    
    global horizon_iteration # nombre de commandes prédites
    global states_ref

    global current_state
    global current_command
    
    global u_ref
    global x_ref
    global t_ref

    global seq_fin

    # initialisation du contexte
    print "[*] Définition d'une suite de primitives de mouvement"
    seq_fin = [maneuvers[0].copy()]
    print "[*] Définition des trajectoires de référence"
    set_state_ref2(State(0,4,0,0,0,0,0,0,0))
    print "[*] Suivi de ces trajectoires par MPC sous gazebo"

    
    # etats intial
    current_state   = State(0,4,0,0,0,0,0,0,0)
    current_command = Command(0,0)
    
    # définition du chemin sur une horizon
    u_ref = get_u_ref(current_state)
    x_ref = get_x_ref(current_state)
    t_ref = get_t_ref(current_state)

    #print "x_ref : ",x_ref
        
    # détermination des prévisions de commande initial
    u_0 = []
    for i in range(0,horizon_iteration) :
        u_0 += [0,0]#[current_command.betaf, current_command.betar]
    
    u_0 = [u_0]
     
    t1 = time.time()
    u_optimized = fmin_cg(fun=f_cost_MPC, x0=u_0).x#, accept_test=mybounds, niter=1).x
    t2 = time.time()
    #erreur.append(u_optimized[0]-u_ref[0])

    print "u optimized      : ",u_optimized
    print "computation time : ",(t2-t1)

    print "f_cost pour une commande nulle (que des 0) : ",f_cost_MPC([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #print "x_ref : ",x_ref
    #print "u_ref : ",u_ref

def test_f_cost() :
    
    global horizon_iteration # nombre de commandes prédites
    global betaMax

    # détermination de la trajectoire de référence
    print "[*] Définition d'une suite de primitives de mouvement"
    seq_fin = [maneuvers[0].copy()]

    print "[*] Définition des trajectoires de référence"
    ref_starting_state = State(0,4,0,0,0,0,0,0,0)
    states_ref = set_state_ref2(ref_starting_state,seq_fin)

    # définition du chemin sur une horizon
    u_ref = get_u_ref(ref_starting_state, states_ref)
    x_ref = get_x_ref(ref_starting_state, states_ref)
    t_ref = get_t_ref(ref_starting_state, states_ref)

    # etats intial
    current_state   = State(0,4,0,0,0,0.1,0,0,0)
    current_command = Command(0,0)

    # calcule les angles de braquages nécessaires au recalage du robot sur la trajectoire de référence
    # utilisation du MPC
    # détermination des prévisions de commande initial
    u_0 = []
    for i in range(0,horizon_iteration-1) :
        u_0 += [-0.01,-0.01]#[current_command.betaf, current_command.betar]
    
    #u_0 = [  0.00000000e+00,   6.05018397e-03,   4.33432624e-03,   3.06946868e-03,   2.19297434e-03,   1.64280083e-03,   1.35806239e-03,   1.27976647e-03,   1.35171937e-03,   8.12823966e-04,   3.79555381e-04,   1.37448875e-04,   2.46174551e-05,  -1.11959116e-05,  -1.09786696e-05,  -2.78596873e-06,  -6.89801389e-08,   0.00000000e+00,   1.00000000e-08]


    u_0 = [u_0]

    # etablissement des contraintes
    #cons = []
    #for i in range(0,len(u_0[0])) :
    #    cons.append((-betaMax,betaMax))
    
    print "f_cost value : ",f_cost_MPC(u_0[0],[current_state,current_command,u_ref,x_ref,t_ref])

def test_without_minimization() :

    current_state = State(0,4,0,0,0,0.1,0,0,0)

    H = get_H_matrix(current_state)
    P = get_P_matrix(1, 10)
    R = get_R_matrix(1, 10)
    G = get_G_matrix(current_state)

    print "H : ",H
    print "P : ",P
    print "R : ",R
    print "G : ",G

if __name__ == '__main__':
    test_f_cost()
