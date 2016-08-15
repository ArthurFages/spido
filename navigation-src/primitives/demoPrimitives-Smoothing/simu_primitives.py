#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
"""
Script permettant de simuler le comportement du SPIDO sur une primitive de mouvement
et d'enregistrer les états résultats (Vpsi,Vx,Vy,psi,x,y,betaf,betar,t) dans un
fichier.
Les résultats obtenus sont affichés pour vérification.
Les paramètres de la primitive de mouvement sont modifiables dans la zone indiqué
à cet effet.

La primitives paramétrée est enregistrée dans le cas d'une rotation vers la droite
et d'une rotation vers la gauche (indépendamment l'une de l'autre).
Les nom de fichier résultants sont de la forme :

    virage_<droite|gauche>_<suffixe utilisateur>
"""


import numpy as np
import pylab as pl

from classes import *
from model import *

import eq_diff as eq
import rospy
import os
import matplotlib.pyplot as plt
import time


################################################################################
###     PARAMETRES A CHANGER POUR SIMULER UNE PRIMITIVE DE MOUVEMENT    ########
################################################################################

angle_rotation      = np.radians(-10) # pour le virage à droite (gauche = opposé)
temps_simulation    = 1 #s
suffixe_primitive   = "court_10"

################################################################################



# repère lié au départ du robot
psi_0   = 122
x_0     = 122
y_0     = 122

# temps simulé
t = 0
t_0 = 0

# etat de la commande <=> valeur des angles de braquage
beta_f = 0
beta_r = 0

# etat actuel du robot
initial_state = State(0,0,0,0,0,0,0,0,0)

# états successifs du robot
states = []

# taille de la simulation
T_discretisation = 0.1

Vx = 4.0
integration_factor  = 3
exponential_factor = 3
thau_rk = 0.01



def simulate(u, current_state,T_discretisation) :

    #global sum_factor
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
    
    b = get_model_b_matrix()
    stable_current_state = current_state.copy()
    
    # commande et états simulés à l'aide du modèle
    x_i     = np.array([stable_current_state.Vy, stable_current_state.Vpsi, stable_current_state.psi, stable_current_state.x, stable_current_state.y])
    u_i     = np.array(u[0])
    
    states = [current_state]

    #print "states[-1] : ",states[-1]

    T = T_discretisation
    t0 = 0

    for i in range(1,len(u)):

        print "Commande : ",u_i

        # enregistement de l'état actuel
        states.append(State(x_i[1],Vx,x_i[0],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1],t0+i*T_discretisation))
        print states[-1].to_string_v()

        syst_CI = np.array([x_i[0],x_i[1],x_i[2],x_i[3],x_i[4],u_i[0],u_i[1]]) 

        #print "syst CI : ",syst_CI

        # simulation du modèle
        t = pl.frange(0,T,thau_rk) 
        x_rk4    = eq.rk4(deriv, syst_CI, t)

        print "x_rk4 : ",x_rk4[-1]

        # enregistrement du nouvel etat
        #for j in range(0,5) :
        #    x_i[j] = x_rk4[-1][j]
        #    print "x_i : ",x_i[j]
        #    print "x_rk4 : ",x_rk4[-1][j]

        x_i = x_rk4[-1]
        #print x_i

        u_i = u[i]
        
        # chargement de la commande prédite à l'itération suivante, et de la commande et trajectoire de référence à l'itération suivante
        X.append(x_i[3])
        Y.append(x_i[4])

    print "Pas de discretisation : ",T_discretisation
                                

    return X, Y, states

        
def record_states(states) :
    # enregistrement de la primitive dans un fichier au format csv
    global nom_primitive
    
    fichier = open(nom_primitive,"a")
    
    for state in states :
        line = state.to_string()+"\n"
        fichier.write(line)
    
    fichier.close()
    
def display_states(states) :
    
    Vx      = []
    Vy      = []
    Vpsi    = []
    psi     = []
    x       = []
    y       = []
    t       = []
    
    for state in states :
        print state.to_string()
        Vx.append(state.Vx)
        Vy.append(state.Vy)
        Vpsi.append(state.Vpsi)
        x.append(state.x)
        y.append(state.y)
        psi.append(state.psi)
        t.append(state.t)
    
    plt.figure("Evolution etat robot")
    plt.subplot(2,2,1)
    plt.title("Trajectoire")
    #plt.plot(t,y)
    #plt.plot(t,x)
    plt.plot(x,y)
    plt.subplot(2,2,2)
    plt.title("Psi au cours du temps")
    plt.plot(t,psi)
    plt.subplot(2,2,3)
    plt.title("Vy au cours du temps")
    plt.plot(t,Vy)
    plt.subplot(2,2,4)
    plt.title("Vpsi au cours du temps")
    plt.plot(t,Vpsi)
    
    #plt.figure("Vx")
    #plt.plot(t,Vx)
    plt.show()
    
def print_info_prim(states) :
    
    print "Dx : "+str(states[-1].x-states[0].x)
    print "Dy : "+str(states[-1].y-states[0].y)
    print "Dpsi : "+str(states[-1].psi-states[0].psi)
    
    
def main() :
    
    global angle_rotation
    global temps_simulation
    global nom_primitive
    global suffixe_primitive
    global states

    global initial_state
    global T_discretisation
    
    global psi_0
    global x_0
    global y_0
    
    # initialisation pour la prochaine primitive
    psi_0   = 122
    x_0     = 122
    y_0     = 122
    states = []
    
    # virage à droite
    print "[*] Virage à droite"
    nom_primitive       = "virage_droite_"+suffixe_primitive
    
    # suppression des fichiers existants
    try :
        os.remove(nom_primitive)
    except :
        print "> Pas de suppression de fichier"

    # génération d'une commande de longueur
    size_simu = temps_simulation / T_discretisation
    u = [[angle_rotation,angle_rotation]]*int(size_simu)

    # simulation de la commande et récupération de la succession d'états
    X, Y, states = simulate(u, initial_state,T_discretisation)
    display_states(states)
    print_info_prim(states)
    record_states(states)
    
    # initialisation pour la prochaine primitive
    psi_0   = 122
    x_0     = 122
    y_0     = 122
    states = []
    
    # virage à gauche
    print "[*] Virage à gauche"
    angle_rotation      = -angle_rotation
    nom_primitive       = "virage_gauche_"+suffixe_primitive
    
    # suppression des fichiers existants
    try :
        os.remove(nom_primitive)
    except :
        print "> Pas de suppression de fichier"

    # génération d'une commande de longueur
    size_simu = temps_simulation / T_discretisation
    u = [[angle_rotation,angle_rotation]]*int(size_simu)
    
    X, Y, states = simulate(u, initial_state,T_discretisation)
    display_states(states)
    print_info_prim(states)
    record_states(states)

if __name__ == '__main__':
    main()
    
    
