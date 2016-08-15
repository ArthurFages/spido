#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

from simu_primitives import *
from parameters import *
import numpy as np

# -- Fonctions utilitaires pour le chargement des primitives -------------
def primitive_from_file(file, id_man) :
    
    global maneuvers
    
    states = []
    with open(file) as f:
        for line in f:
            state_value = line.split(',')
            states.append(State(state_value[3], state_value[4], state_value[5], state_value[6], state_value[7], state_value[8], state_value[1], state_value[2], state_value[0]))
    
    #display_states(states)
    maneuvers[id_man].id_maneuver= id_man
    maneuvers[id_man].states     = states
    #print "len state maneuver ",id_man," : ",len(maneuvers[id_man].states)
    maneuvers[id_man].thau       = float(states[-1].t) - float(states[0].t)
    maneuvers[id_man].delta_psi  = float(states[-1].psi) - float(states[0].psi)
    #print "state[0].psi : ",float(states[0].psi)
    maneuvers[id_man].delta_x    = float(states[-1].x) - float(states[0].x)
    maneuvers[id_man].delta_y    = float(states[-1].y) - float(states[0].y)
    

def set_maneuver_0(horizon_iteration,T_discretisation) :

    global maneuvers
    global Vx

    states_ref = []

    #horizon_iteration = horizon_iteration*2 # pour être sur que l'on a de la marge pour les tests MPC
                    
    # chargement des maneuvres depuis les fichiers correspondants

    # tout droit (maneuver d'indice 0)
    states = []
    size = horizon_iteration # on fixe la longueur de la primitive sur la longueur de la prédiction
    t = 0
    tf = (horizon_iteration+1)*T_discretisation # Vx m.s-1 => Vx m en 1s
    psi = np.radians(0)
    delta_x = tf*Vx*np.cos(psi)
    delta_y = tf*Vx*np.sin(psi)

    for i in range(0,size) :
        # self, Vpsi, Vx, Vy, psi, x, y, beta_f, beta_r, t=None) :
        states.append(State(0,Vx,0,0,(delta_x/size)*i,(delta_y/size)*i,0,0,(tf/size)*i))
    maneuvers[0].states     = states
    #print "len state maneuver 0 : ",len(maneuvers[0].states)
    maneuvers[0].thau       = tf
    maneuvers[0].delta_psi  = float(states[-1].psi) - float(states[0].psi)
    #print "state[0].psi : ",float(states[0].psi)
    maneuvers[0].delta_x    = float(states[-1].x) - float(states[0].x)
    maneuvers[0].delta_y    = float(states[-1].y) - float(states[0].y)

# -----------------------------------------------------------------------------------



def set_primitives(T_discretisation) :
    temps_simulation_tab = [2,5]
    angle_rotation_tab = [-10,-20]

    files_names = record_primitives_in_files(T_discretisation, temps_simulation_tab, angle_rotation_tab)
    load_primitives_from_files(files_names)


def load_primitives_from_files(files_names) :

    global maneuvers

    i_man = 0

    for file_name in files_names :
        maneuvers.append(Maneuver(0,0,0,0,0,0,0,file_name))
        primitive_from_file(file_name,i_man)
        i_man += 1


