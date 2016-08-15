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
from simulation import *

import eq_diff as eq
import rospy
import os
import matplotlib.pyplot as plt
import time



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

# états successifs du robot
states = []


integration_factor  = 3
exponential_factor = 3
thau_rk = 0.01

        
def record_states(states, nom_primitive) :
    # enregistrement de la primitive dans un fichier au format csv
    #global nom_primitive
    
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

"""
angle de rotation : angle de rotation maximal de la commande
show : si vrai affiche graphiquement la commande générée
"""
def get_gaussian_command(angle_rotation, temps_simulation, horizon_iteration, show = False) :

    # la commande lors d'un virage suit une fonction gaussienne
    m = horizon_iteration/3       # "position de la gaussienne"
    sigma = horizon_iteration/15   # "grosseur" gaussienne (real gaussian have curves)
    angle_braquage = lambda x: angle_rotation*np.exp((-1/2)*(((x-m)**2/(2*sigma**2))))

    u = []
    a_tab = []
    b_tab = []
    #for i in range(0,10) :
    #    u.append([0,0])
    for i in range(0,int(horizon_iteration)) :
        a = angle_braquage(i)
        a_tab.append(a)
        b = -angle_braquage(i)
        b_tab.append(b)
        u.append([a,b])

    if show :
        plt.plot(range(0,len(a_tab)),a_tab)
        plt.plot(range(0,len(b_tab)),b_tab)
        plt.show()

    return u
    
    
def main(angle_rotation, temps_simulation, suffixe_primitive, initial_state, T_discretisation) :
    
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
    ## on fait en sorte que l'angle de départ soit nul, tout comme l'angle terminal
    ## pour permettre l'enchainement correct de primitives de mouvement
    #u = [[angle_rotation,-angle_rotation]]*int(size_simu)
    u = get_gaussian_command(angle_rotation, temps_simulation, size_simu)

    # simulation de la commande et récupération de la succession d'états
    X, Y, states = simulate(u, initial_state,T_discretisation)
    #display_states(states)
    #print_info_prim(states)
    record_states(states,nom_primitive)
    
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
    ## on fait en sorte que l'angle de départ soit nul, tout comme l'angle terminal
    ## pour permettre l'enchainement correct de primitives de mouvement
    #u = [[angle_rotation,-angle_rotation]]*int(size_simu)
    u = get_gaussian_command(angle_rotation, temps_simulation, size_simu)
    
    X, Y, states = simulate(u, initial_state,T_discretisation)
    #display_states(states)
    #print_info_prim(states)
    record_states(states,nom_primitive)

def record_primitives_in_files(T_discretisation, temps_simulation_tab, angle_rotation_tab) :

    # etat initial du robot
    initial_state = State(0,0,0,0,0,0,0,0,0) # TODO : passer en argument pour adapter la primitive au contexte
    files_names = []

    for temps_simulation in temps_simulation_tab :

        suffixe_primitive = "long"
        if temps_simulation < 3 :
            suffixe_primitive   = "court"

        for angle in angle_rotation_tab :

            angle_rotation = np.radians(angle)

            main(angle_rotation, temps_simulation, suffixe_primitive+str(angle), initial_state, T_discretisation)

            files_names.append("virage_droite_"+suffixe_primitive+str(angle))
            files_names.append("virage_gauche_"+suffixe_primitive+str(angle))
            


    print "files names : ",files_names
    return files_names

    
    
