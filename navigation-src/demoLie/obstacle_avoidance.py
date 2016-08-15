#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate
import rosbag
import std_msgs.msg
from nav_msgs.msg import Odometry
import sys


from classes import Node2
from simu_primitives import *
from simulation import *

# recording path for rosbag
rosbag_record_path = "/home/spido/catkin_ws/rosbag/"

def display_path(states, obstacles) :

    Vy = []
    Vpsi = []
    psi = []
    x = []
    y = []
    u = []
    t = []

    for state in states :
        Vy.append(state.Vy)
        Vpsi.append(state.Vpsi)
        psi.append(state.psi)
        x.append(state.x)
        y.append(state.y)
        u.append(state.beta_f)
        t.append(state.t)

    plt.figure("Trajectoire resultante")
    fig = plt.gcf()
    ax = fig.gca()
    plt.scatter(x,y,s=100, c='y')
    for obstacle in obstacles :
        #plt.scatter(obstacle.x, obstacle.y, obstacle.r*1e4)
        ax.add_artist(plt.Circle((obstacle.x, obstacle.y), obstacle.r, color='g'))
    #plt.show()

#def dist_between_two_states(state1, state2) :
#    return np.sqrt((state2.x-state1.x)**2 + (state2.y-state1.y)**2)

def dist_between_two_points(point1, point2) :
    return np.sqrt((point2.x-point1.x)**2 + (point2.y-point1.y)**2)

"""
first one : le plus proche de l'état final
last one : le moins proche de l'état final
"""
def order_states(states, final_state) :

    ordered_states = []

    while len(states) > 0 :
        best_state = None
        dist_min = 1e122
        for state in states :
            dist = dist_between_two_points(final_state, state[0])
            if  dist < dist_min :
                dist_min = dist
                best_state = state 
        states.remove(best_state)
        ordered_states.append(best_state)

    return ordered_states

def get_path_from_node(node) :

    states_path     = [] 
    command_path    = []
    current_node = node

    while current_node <> None :
        states_path.append(current_node.state)
        command_path += current_node.u
        current_node = current_node.prec

    states_path.reverse()
    command_path.reverse()

    return states_path, command_path


"""
Indique si une primitive de mouvement évolue dans la zone d'action d'un obstacle 

TODO : utiliser une carte de cout (typiquement celle de gmapping dans ROS)

"""
def primitive_in_obstacle_zone(X,Y,obstacles,display_collision=False) :

    for i in range(0,len(X)) :
        for obstacle in obstacles :
            if dist_between_two_points(Point(X[i],Y[i]),Point(obstacle.x, obstacle.y)) <= obstacle.r :
                if display_collision :
                    print "Collision : (",X[i],",",Y[i],")"
                return True
            # ELSE : la primitive ne passe pas dans la zone d'action de l'obstacle courant
    return False

def compute_ref_path(starting_state, ending_state, obstacles, T_discretisation, temps_simulation_primitives) :

    

    

    start_node      = Node2(starting_state, None, [], 0, []) # depending on starting_state
    end_node        = Node2(ending_state, None, [], 0, [])
    current_node    = start_node

    primitives_angles 	= map(np.radians,[0,3,-3,5,-5,10,-10,15,-15,20,-20,25,-25])#,30,-30,40,-40])
    primitives_command 	= []

    
    horizon_iteration = temps_simulation_primitives/T_discretisation
    print "[*] Horizon itération : ",horizon_iteration

    rayon_ending_zone = 4

    for angle in primitives_angles :
        primitives_command.append(get_gaussian_command(angle, temps_simulation_primitives, horizon_iteration, True))

    end_algo = False

    print "[*] Affichage de la tête des primitives depuis la position de départ"
    plt.figure("Primitives depuis la position de départ")
    i = 0 # indice de référencement des angles de primitive utilisés
    for command in primitives_command :
        X,Y,states = simulate(command, current_node.state,T_discretisation)
        plt.plot(X,Y,label="$"+str(primitives_angles[i])+"rad$")
        plt.axis("equal")
        i += 1
    plt.legend()
    plt.show()

    t1 = time.time()

    print "[compute ref path] > ",
    while dist_between_two_points(end_node.state, current_node.state) > rayon_ending_zone and not end_algo :

        sys.stdout.write('.')
        sys.stdout.flush()

        #print "Current node position : ",current_node.to_string()
        #states_path, command_path = get_path_from_node(current_node)
        #display_states(states_path)
        #print "Command path : ",command_path

        if len(current_node.fol) == 0 :
            # première fois qu'on considère l'état courant

            # calcul des primitives possibles depuis l'état courant
            # et identification des états finaux résultants de la simulation
            possible_states_from_current = []
            #plt.figure("primitives")
            for command in primitives_command :
                #print "> Command primitive : ",command
                X,Y,states = simulate(command, current_node.state,T_discretisation)
                #plt.plot(X,Y)
                #plt.axis("equal")
        
                # élimination des états dans les zones ou le robot ne peut pas aller <=> zones d'influence des obstacles
                if not primitive_in_obstacle_zone(X,Y,obstacles) :
                    possible_states_from_current.append([states[-1],command])
                # ELSE : la primitive n'est pas sauvegardée     

            #plt.show()

            if len(possible_states_from_current) > 0 :

                # classement des états restants en fonction de leur proximité avec l'état final
                ordered_possible_states_from_current = order_states(possible_states_from_current,end_node.state)

                # mise à jour du noeud courant
                current_node.fol    = ordered_possible_states_from_current
                current_node.id_fol = 0

                # détermination du noeud suivant
                current_node = Node2(current_node.fol[current_node.id_fol][0], current_node, [], 0, current_node.fol[current_node.id_fol][1])

            else :

                # aucun chemin viable n'est possible depuis l'état courant
                if current_node.prec == None :
                    # l'algo ne peut être mené à terme
                    print "[!] Aucun chemin ne peut être calculé depuis l'état courant, déso mec déso"
                    end_algo = True
                else :
                    # on remonte à l'état précédent
                    current_node = current_node.prec

        else :
            # l'état courant est issus d'un "retour en arrière"
            current_node.id_fol += 1
            if current_node.id_fol < len(current_node.fol) :
                # il reste des états  non explorés depuis l'état courant
                current_node = Node2(current_node.fol[current_node.id_fol][0], current_node, [], 0, current_node.fol[current_node.id_fol][1])
            else :
                # il ne reste pas d'état à explorer
                if current_node.prec == None :
                    # l'algo ne peut être mené à terme
                    print "[!] Aucun chemin ne peut être calculé depuis l'état courant, déso mec déso"
                    end_algo = True
                else :
                    # on remonte à l'état précédent
                    current_node = current_node.prec
    print ""

    states_path, command_path = get_path_from_node(current_node)

    t2 = time.time()

    return states_path, command_path, (t2-t1)

def get_xy_path_from_states(states) :

    x = []
    y = []
    for state in states :
        x.append(state.x)
        y.append(state.y)
    return x, y

"""
Attention : les vitesses Vx, Vy et Vpsi sont exprimés dans le repère global !
"""
def get_tab_states(states) :

    x       = []
    y       = []
    psi     = []
    Vx      = []
    Vy      = []
    Vpsi    = []

    for state in states :
        x.append(state.x)
        y.append(state.y)
        psi.append(state.psi)
        Vx.append(state.Vx*np.cos(state.psi) - state.Vy*np.sin(state.psi))
        Vy.append(state.Vx*np.sin(state.psi) + state.Vy*np.cos(state.psi))
        Vpsi.append(state.Vpsi)

    return x, y, psi, Vx, Vy, Vpsi

def interpolate_states(states) :

    x, y, psi, Vx, Vy, Vpsi = map(np.array,get_tab_states(states))

    t = np.zeros(x.shape)
    nt = np.linspace(0, 1, 100)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]
    x2 = scipy.interpolate.spline(t, x, nt)
    y2 = scipy.interpolate.spline(t, y, nt)
    psi2 = scipy.interpolate.spline(t, psi, nt)
    Vx2 = scipy.interpolate.spline(t, Vx, nt)
    Vy2 = scipy.interpolate.spline(t, Vy, nt)
    Vpsi2 = scipy.interpolate.spline(t, Vpsi, nt)

    return x, y, psi, Vx, Vy, Vpsi

def interpolate(x,y) :

    x = np.array(x)
    y = np.array(y)
    t = np.zeros(x.shape)
    nt = np.linspace(0, 1, 100)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]
    x2 = scipy.interpolate.spline(t, x, nt)
    y2 = scipy.interpolate.spline(t, y, nt)

    psi2 = []
    for i in range(0,len(x2)-1) :
        psi2.append(np.arctan2(y2[i+1]-y2[i],x2[i+1]-x2[i]))
    psi2.append(psi2[-1]) # cas particulier du dernier psi : on part du principe qu'il reste cst

    return x2, y2, psi2

if __name__ == '__main__':

    #
    # WARNINGUE : si temps_simulation_primitives trop élevé part rapport au T_discretisation il y a 
    #             dissenssion entre les états resultant des primitives et la trajectoire simulée avec 
    #             les consignes résultantes
    #
    T_discretisation                = 2
    temps_simulation_primitives     = 8

    starting_state  = State(0,1,0,0,0,0,0,0)
    print "Starting state : ",starting_state.to_string_v()
    ending_state    = State(0,1,0,0,40,40,0,0)
    print "Ending state : ",ending_state.to_string_v()
    obstacles       = [Obstacle(10,10,4),Obstacle(20,30,4),Obstacle(10,5,5),Obstacle(32,30,5),Obstacle(33,10,3)]

    
    states_path, command_path, t = compute_ref_path(starting_state, ending_state, obstacles, T_discretisation, temps_simulation_primitives)
    
    display_path(states_path,obstacles)

    x, y            = get_xy_path_from_states(states_path) 
    x1, y1, psi1    = interpolate(x,y)
    #x2, y2, psi2, Vx2, Vy2, Vpsi2 = interpolate_states(states_path)

    #plt.plot(x,y,c='b',label="$Calculs$")
    plt.plot(x1,y1,c='b',label="$Interpolation$")
    #plt.plot(x2,y2,c='g',label="$Interpolation2$")
    plt.axis("equal")
    plt.legend()
    plt.show()
    print "[*] Temps de calcul : ",t,"s"

    plt.figure("Variation psi")
    plt.plot(range(0,len(psi1)),psi1)
    plt.show()

    #plt.figure("Variation Vx")
    #plt.plot(range(0,len(Vx2)),Vx2)
    #plt.show()

    #plt.figure("Variation Vy")
    #plt.plot(range(0,len(Vy2)),Vy2)
    #plt.show()

    #plt.figure("Variation Vpsi")
    #plt.plot(range(0,len(Vpsi2)),Vpsi2)
    #plt.show()

    # enregistrement de la trajectoire dans un rosbag
    rospy.init_node('spido_primitives_path_recording')
    print "[*] Node created"

    # bags for data visualisation
    ref_path_bag   = rosbag.Bag(str(rosbag_record_path)+'ref_path_primitives.bag', 'w')
    print "[*] Bag correctly created"
    
    
    try :
        for i in range(0,len(x1)) :
            
            odom_result = Odometry()
            
            # position x,y
            odom_result.pose.pose.position.x = x1[i]
            odom_result.pose.pose.position.y = y1[i]
            
            # position angulaire
            odom_result.pose.pose.orientation.x = 0
            odom_result.pose.pose.orientation.y = 0
            odom_result.pose.pose.orientation.z = np.sin(psi1[i]/2)
            odom_result.pose.pose.orientation.w = np.cos(psi1[i]/2)
            
            # vitesses
            #odom_result.twist.twist.linear.x    = 0#Vx2[i]
            #odom_result.twist.twist.linear.y    = 0#Vy2[i]
            #odom_result.twist.twist.angular.z   = 0#Vpsi2[i]

            ref_path_bag.write('odom', odom_result)
    
    except :
        print "[!] Erreur écriture data dans rosbag : ref_path_primitives"
        
    finally :
        ref_path_bag.close()
        print "[*] Bag closed"


    #X,Y,states = simulate(command_path, starting_state, T_discretisation)

    #primitive_in_obstacle_zone(X, Y, obstacles, True)

    #plt.figure("Simulation des angles de braquage consigne générés")
    #display_path(states,obstacles)
    
    #plt.plot(X,Y,c='b',label="$Simulation post algo$")
    #plt.axis("equal")
    #plt.legend()
    
    #plt.figure("Evolution de psi_ref")
    #plt.plot(range(0,len(psi1)),psi1)
    #plt.show()



