#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import rospy
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from pylab import *
import time

from scipy.optimize import basinhopping
from pyOpt import *

from math import fabs
from scipy.integrate import quad
from geometry_msgs.msg import Pose2D
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from random import randint
from mpl_toolkits.mplot3d import Axes3D


# import usefull classes
from parameters import *
from classes import *
from model import *

# reccursion setup
sys.setrecursionlimit(reccursion_limit)
        
# ========================== TRAJECTORY SMOOTHING =========================================

############ All credits to https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/

def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def DistancePointLine (px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine

############ End credits

"""
Distance d'un point à un segment
a, b : bornes du segment
point : point dont on va calculer la distance au segment [a,b]
"""
def dist(a,b,point) :
    return DistancePointLine(point.x, point.y, a[0], a[1], b[0], b[1])
    
"""
point issu de la projection d'un point sur un segment
"""
def project(point, a, b) :
    
    x_min = y_min = dist_min = 1e122
    
    size = 1000
    
    dx = (b.x - a.x)/size
    dy = (b.y - a.y)/size
    
    for i in range(0,size) :
        
        x = a.x + i*dx
        y = a.y + i*dy
        
        dist = disteuclidean(Point(x,y),point)
        
        if dist < dist_min :
            dist_min    = dist
            x_min       = x
            y_min       = y 
        
    return Point(x_min,y_min)

"""
Position : psi, x, y
"""
def new_position_from_maneuver(node, maneuver) :
    
    psi = node.theta + maneuver.delta_psi
    x   = node.pt.x + maneuver.delta_x
    y   = node.pt.y + maneuver.delta_y
    
    return [psi,x,y]
    

def dist_nearest(state,w) :
    
    dist_nearest = 1e122
    
    X = w[0]
    Y = w[1]
    
    for i in range(0,len(X)) :
        dist = np.sqrt((X[i]-float(state.x))**2+(Y[i]-float(state.y))**2)
        if dist < dist_nearest :
            dist_nearest = dist
    
    return dist_nearest
    
"""
Distance euclidienne entre deux points
"""
def disteuclidean(a, b):
    return np.sqrt((a.x-b.x)**2+(a.y-b.y)**2)
    
"""
alpha : angle de rotation
L : liste de points dont on souhaite faire la rotation d'angle alpha
"""
def rotation_2D(L,alpha) :
    
    result = []
    R = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
    for vector in L :
        result.append(R.dot(np.array(vector)))
    return result

def get_oriented_maneuver(man,theta) :

    m = man.copy()
    
    # Vx, Vy, Vpsi restent identiques (car liés au repère du robot)
    # psi, x et y affectés par l'orientation du SPIDO 
    oriented_states = []
    for state in m.states :
        state.psi = float(state.psi) + theta
        rot = rotation_2D([[float(state.x),float(state.y)]],theta)[0]
        state.x = rot[0]
        state.y = rot[1]
        
        #x.append(state.x)
        #y.append(state.y)
        
    m.delta_x = m.states[-1].x - m.states[0].x
    m.delta_y = m.states[-1].y - m.states[0].y
    m.delta_psi = m.states[-1].psi - m.states[0].psi
    
    return m
    
"""
position : [psi,x,y]
"""
def new_node_from_position(position, m, w) :

    target_zone_radius = 2
    return Node(0,0,0,position[0],Point(position[1],position[2]),0,0,np.sqrt((w[0][-1]-position[1])**2+(w[1][-1]-position[2])**2) < target_zone_radius)

"""
Retourne l'indice de la trajectoire par méthode des champs de potentiel

"""
def get_s(node) :
    
    global pot_field_trajectory
    
    min = 1e122
    s = 0
    
    for i in range(0,len(pot_field_trajectory[0])) :
        val = np.sqrt((pot_field_trajectory[0][i]-node.pt.x)**2 + (pot_field_trajectory[1][i]-node.pt.y)**2)
        if val < min :
            min = val
            s = i
    
    return s

def display_current_path(L, initial_state) :
    i_traj = 0
    for node in L : 
        #get and return path from root node to target node gbest using pr ed(g)
        path = []
        cur_node = node
        while cur_node.prec <> None :
            path.append(cur_node)
            cur_node = cur_node.prec
            
        # détermination succession de primitives
        path.reverse()
        
        #print "reversed path : ",path
        primitives  = []
        states_succession = []
        for node in path :
            if node.i_man >= 0 :
                primitives.append(node.i_man)
                states_succession.append(node.man)
    
        plt.figure("Trajectoire "+str(i_traj))
        plot_computed_motions(initial_state,states_succession,False)
        i_traj += 1

"""
 w : signal de référence, issu de la méthode des champs de potentiel
 M : ensemble de primitive de mouvement
 initial_state : position initiale du SPIDO
"""
def approximate(initial_state, pot_field_trajectory,show_intermediate_trajectory = False) :#w, M) :
    
    #global pot_field_trajectory #w [X_pf,Y_pf]
    global maneuvers # M
    global time_limit
    #global states_succession
    
    w = pot_field_trajectory

    #init empty tree G
    G = []
    #init empty target nodes list L
    L = []
    #add root node gr oot to G with pr ed(gr oot ) ← −1, c(gr oot ) ← 0, h(gr oot ) ← 0, pt (gr oot ) ←t0 , θdisc (gr oot ) = θdisc (w0 w1 )
    # prec, heuristic, cost, theta, point, man, isTarget=False) :
    root_node = Node(None, 0, 0, initial_state.psi, Point(initial_state.x,initial_state.y),-1,-1,False,True)
    G.append(root_node)
    
    time_0  = time.time()
    time_t  = time.time()
    
    alpha = 100
    
    #while G has nodes and time limit not exceeded do
    while G and len(L)==0 and (time_t-time_0) < time_limit :

        #select best node gbest of G regarding f (g) = α ∗ h(g) + c(g)
        gbest = None
        f_min = 1e20
        for node in G :
            f = alpha*node.h + node.c
            if f < fmin :
                f_min = f
                gbest = node
        
        #expand gbest using m ∈ M, where θdisc (m) = θdisc (gbest )
        for man in maneuvers :
            
            print "test maneuver : ",man.id_maneuver
            
            m = get_oriented_maneuver(man,gbest.theta)
            gi = new_node_from_position(new_position_from_maneuver(gbest,m),m,w)
            
            print "[gi] ",gi.pt.to_string()
            
            #get cost c(gi ), heuristic h(gi ), location pt (gi ) and orientation θdisc (gi ) for successors
            gi.prec     = gbest
            gi.man      = m.copy()
            gi.i_man    = man.id_maneuver
            gi.c        = cost(gi,w,m)
            
            #add valid (c(gi ) < ∞) successors gi to G (with pr ed(gi ) ← gbest )
            if gi.c < 1e122 :
                gi.h      = heuristic(gi,w,get_s(gi))
                print "-> Succeed"
                G.append(gi)
                print "Ajout du node dans G : ",G
                
            # if gi is target node, add to target node list L
            if gi.isTarget :
                L.append(gi)

        #remove gbest from G
        G.remove(gbest)
        
        print "_________________________________"
        print "G : ",G
        print "L : ",L
        #display_current_path(G, initial_state)
        
        #adapt α for next iteration
        
        time_t = time.time()
        
    #select best target node gbest from target list L
    
    if len(L)==0 :
        print "[*] Pas de lissage possible avec les primitives données"
        return

    # affichage de tout les chemins calculés
    #if show_intermediate_trajectory :
        #display_current_path(L, initial_state)

    gbest = None
    f_min = 1e20
    for node in L :
        #print "c : ",node.c
        #print "h : ",node.h
        
        f = alpha*node.h + node.c
        if f < fmin :
            f_min = f
            gbest = node
    
    #print "gbest : ",gbest
    
    #get and return path from root node to target node gbest using pr ed(g)
    path = []
    cur_node = gbest
    while cur_node.prec <> None :
        path.append(cur_node)
        cur_node = cur_node.prec

    
    #print "path : ",path
        
    # détermination succession de primitives
    path.reverse()
    
    #print "reversed path : ",path
    primitives  = []
    states_succession = []
    for node in path :
        if node.i_man >= 0 :
            primitives.append(node.i_man)
            states_succession.append(node.man)

    print "primitives : ",primitives
    
    primitives_result = []
    for i in primitives :
        primitives_result.append(maneuvers[i])
    
    print "state_succession : ",states_succession

    return primitives_result, states_succession

"""
Cout de la sortie éventuelle du "tunnel" définissant les trajectoires possibles
Prise en compte de la "zone d'effet" des obstacles
"""
def cost(g, w, m) :
    
    global cost_tolerance
    global obstacle_tolerance
    
    end_cost = False
    
    #sum ← c( pr ed(g))
    if not g.isRoot :
        #print g
        sum = g.prec.c

        for state in m.states :
            
            # application de la maneuvre à la position actuelle
            cur_state = state
            cur_state.x += g.pt.x
            cur_state.y += g.pt.y
            
            # prise en compte de la "zone d'effet" des obstacles
            for obstacle in obstacles :
                if (obstacle.x + obstacle_tolerance > cur_state.x 
                    and obstacle.x - obstacle_tolerance < cur_state.x 
                    and obstacle.y + obstacle_tolerance > cur_state.y
                    and obstacle.y - obstacle_tolerance < cur_state.y) :
                    sum = 1e122
                    break
            
            if sum >= 1e122 :
                break

            #if distnearest ( ptm i , w) > tolerance then
            if dist_nearest(cur_state,w) > cost_tolerance :
                #sum ← ∞
                sum = 1e122
                break
            else :
                #sum ← sum + distnear est ( ptm i , w)
                sum += dist_nearest(cur_state,w)
                
    else :
        sum = 0
    #c(g) ← sum
    return sum

"""
Cout de l'écartement (distance et angle) de la trajectoire de référence
"""
def heuristic(g, w, s) :
    
    #near est ← 0
    nearest = 0
    #distmin ← ∞
    distmin = 1e122

    h = 0

    #for i ← s, |w| − 1 do
    for i in range(s,len(w[0])-1) :
        #if dist (wi , wi+1 , pt (g)) < distmin then
        distance = dist([w[0][i],w[1][i]],[w[0][i+1],w[1][i+1]],g.pt)
        if distance < distmin :
            #pt (g) :=point of node g
            #distmin ← dist (wi , wi+1 , p(g))
            distmin = distance
            #min distance of p(g) to segment wi wi+1
            #near est ← i
            nearest = i

    #pt pr ojection ← pr oject ( p(g), wnear est , wnear est+1 )
    pt_projection = project(g.pt,Point(w[0][nearest],w[1][nearest]),Point(w[0][nearest+1],w[1][nearest+1]))
    #project p(g) onto segment
    #sum ← disteuclidean ( pt pr ojection , wnear est+1 )
    sum = disteuclidean(pt_projection,Point(w[0][nearest+1],w[1][nearest+1]))
    
    #for i ← near est + 1, |w| − 1 do
    for i in range(nearest+1, len(w[0])-1):
        #sum lengths of remaining segments
        #sum ← sum + disteuclidean (wi , wi+1 )
        sum += disteuclidean(Point(w[0][i],w[1][i]),Point(w[0][i+1],w[1][i+1]))

    #δθ ← |θc (g) − angle(wnear est , wnear est+1 )|
    a = w[0][nearest+1] - w[0][nearest]
    b = w[1][nearest+1] - w[1][nearest]
    
    dTheta = abs(g.theta - math.atan2(b,a))
    beta = 10
    
    #angle difference to current segment
    #h(g) ← distmin + sum + β ∗ δθ
    #β : weighting factor for angle difference
    h = distmin + sum + beta*dTheta
    
    return h



# ========================== TRAJECTORY PLANNING ===========================================

    
################################################################################
# Fonctions de génération d'état
################################################################################


# i_man : indice de la maneuver à effectuer
# i_trim : indice de la trim résultant de la maneuver (trim à atteindre <=> trim a t+)
# /!\ deltas exprimés dans le repère du robot =/= ACC2012_GRAY
"""
[!] : les temps ne sont pas instanciés
"""
def new_state_from_maneuver(current_state, i_trim, i_man) :
    
    global maneuvers
    global trims
    
    #print "[new state from maneuvers] cur state : ",current_state,", trim ",i_trim," i_man",i_man
    
    
    R = np.matrix([ [np.cos(current_state.psi), -np.sin(current_state.psi), 0],
                    [np.sin(current_state.psi), np.cos(current_state.psi), 0],
                    [0, 0, 1]])
    
    E = [   maneuvers[i_man].delta_x,
            maneuvers[i_man].delta_y,
            maneuvers[i_man].delta_psi]
            
    add_state = R.dot(E)
    
    return State(   trims[i_trim].Vpsi,
                    trims[i_trim].Vx,
                    trims[i_trim].Vy,
                    current_state.psi + add_state.item(2),
                    current_state.x   + add_state.item(0),
                    current_state.y   + add_state.item(1),
                    trims[i_trim].beta_f,
                    trims[i_trim].beta_r
                    )
                    
"""
[!] : les temps ne sont pas instanciés
"""
def new_state_from_trim(current_state, i_trim, thau_q) :

    global trims
    
    #dt = 0.01
    new_state = current_state
    
    E = [   trims[i_trim].Vx,
            trims[i_trim].Vy,
            trims[i_trim].Vpsi]
    
    R = np.matrix([ [np.cos(new_state.psi), -np.sin(new_state.psi), 0],
                        [np.sin(new_state.psi), np.cos(new_state.psi), 0],
                        [0, 0, 1]])
     
    add_state = R.dot(E)*thau_q
    
    new_state = State(  trims[i_trim].Vpsi,
                        trims[i_trim].Vx,
                        trims[i_trim].Vy,
                        new_state.psi + add_state.item(2),
                        new_state.x   + add_state.item(0),
                        new_state.y   + add_state.item(1),
                        trims[i_trim].beta_f,
                        trims[i_trim].beta_r)
                            
    return new_state


##
## Champs de potentiel
##
    
def get_potential(ref_position, current_position) :
    
    global obstacles
    
    k_att = 0.1
    k_rep = 500
    rho_0 = 10
    
    rho = np.sqrt((current_position.x-ref_position.x)**2 + (current_position.y-ref_position.y)**2)
    
    # détermination potentiel attractif (target)
    # quadratique
    U_att = (1/2)*k_att*(rho**2)

    # détermination potentiel répulsif (obstacles)
    U_rep = 0
    for i in range(0,len(obstacles)) :
        rho = np.sqrt((current_position.x-obstacles[i].x)**2 + (current_position.y-obstacles[i].y)**2)
        if rho < rho_0 and rho != 0:
            # l'obstacle est à considérer
            U_rep = U_rep + (1/2)*k_rep*(1/rho-1/rho_0)**2

        # else : l'obstacle est trop loin pour qu'on s'en occupe
        
    return U_att + U_rep

def get_state_from_potential_field(ref_position, current_position) :
    
    global k
    global k2
    
    #global delta_t_simu
    
    # initial state
    psi_pot_min     = np.radians(-180-k)
    x_pot_min       = current_position.x + np.cos(psi_pot_min)
    y_pot_min       = current_position.y + np.sin(psi_pot_min)
    min_potential   = get_potential(ref_position, Point(x_pot_min,y_pot_min))
    
    for psi in pl.frange(-180,180,k) :
         
        x_pot       = current_position.x + np.cos(np.radians(psi))
        y_pot       = current_position.y + np.sin(np.radians(psi))
        potential   = get_potential(ref_position, Point(x_pot,y_pot))
        
        if potential<min_potential :
            min_potential   = potential
            psi_pot_min     = np.radians(psi)
    
    # variation temporelle
    #dt = abs(delta_t_simu)
    
    #if dt==0:
    #    dt=1
    
    # variation position
    dx  = k2*np.cos(psi_pot_min)
    dy  = k2*np.sin(psi_pot_min)
    
    #print "psi_pot : ",psi_pot_min," - dx : ",dx," - dy : ",dy
    psi_pot.append(psi_pot_min)
    
    return Output(  current_position.x+dx,
                    current_position.y+dy,
                    psi_pot_min,
                    0,#dy/dt,
                    0)#(psi_pot_min-current_position.psi)/dt)
        
def set_potential_field_trajectory(spido_state,target) :
    
    print "[*] Determination trajectoire avec champs de potentiels"
    
    global obstacles
    global trajectory
    global tolerance
    
    global pot_field_trajectory
    
    X_pf = []
    Y_pf = []
    x_pf = spido_state.x
    y_pf = spido_state.y
    new_pf_state = spido_state
    
    print "Target       : ",target.to_string()
    print "Obstacles    : ",obstacles
    
    fig = plt.figure("Champs de potentiel")
    
    # champs de potentiel
    ax = Axes3D(fig)
    
    #obstacles
    for i in range(0,len(obstacles)):
        ax.scatter(obstacles[i].x,obstacles[i].y,0)
    
    # target
    for i in range(0,len(trajectory)):
        ax.scatter([trajectory[i].x],[trajectory[i].y],0)
    X_map = np.arange(-5, 60, 1)
    Y_map = np.arange(-5, 60, 1)
    i,j = 0,0
    Z = []
    for y in Y_map :
        Z_i_j = []
        for x in X_map :
            Z_i_j.append(get_potential(trajectory[0],Point(x,y)))
        Z.append(Z_i_j)
    X_map, Y_map = np.meshgrid(X_map, Y_map)
    ax.plot_wireframe(X_map, Y_map, Z, rstride=1, cstride=1)
    
    plt.show()
    
    while(x_pf>=target.x+tolerance or x_pf<=target.x-tolerance
            or y_pf>=target.y+tolerance or y_pf<=target.y-tolerance) :
        new_pf_state = get_state_from_potential_field(target,new_pf_state)
        X_pf.append(new_pf_state.x)
        Y_pf.append(new_pf_state.y)
        x_pf = new_pf_state.x
        y_pf = new_pf_state.y
        
        #print "x : ",x_pf," - y : ",y_pf
    
    #pot_field_trajectory = interp1d(x1, y1, kind='linear',  fill_value=0)
    pot_field_trajectory = [X_pf,Y_pf]
    
    # trajectoire
    print "> Display trajectory"
    
    fig = plt.figure("Champs de potentiel")
    
    # champs de potentiel
    ax = Axes3D(fig)
    
    # trajectoire
    ax.scatter(X_pf,Y_pf)
    
    #obstacles
    for i in range(0,len(obstacles)):
        ax.scatter(obstacles[i].x,obstacles[i].y,0)
    
    # target
    for i in range(0,len(trajectory)):
        ax.scatter([trajectory[i].x],[trajectory[i].y],0)
    X_map = np.arange(-5, 60, 1)
    Y_map = np.arange(-5, 60, 1)
    i,j = 0,0
    Z = []
    for y in Y_map :
        Z_i_j = []
        for x in X_map :
            Z_i_j.append(get_potential(trajectory[0],Point(x,y)))
        Z.append(Z_i_j)
    X_map, Y_map = np.meshgrid(X_map, Y_map)
    ax.plot_wireframe(X_map, Y_map, Z, rstride=1, cstride=1)
    
   
    plt.show()

    return pot_field_trajectory
    

###################################################################################################################################
###################################################################################################################################

    
def display_available_primitives() :
    
    global maneuvers
    
    for man in maneuvers :
        x = []
        y = []
        for state in man.states :
            x.append(state.x)
            y.append(state.y)
        plt.plot(x,y)
        
    plt.show()

indice_graph = 0

"""
x et y supposé de même dimension
l = largeur du couloir
"""
def get_path_lane(x,y,l) :

    lane_x_h = []
    lane_y_h = []
    lane_x_l = []
    lane_y_l = []

    for i in range(1,len(x)) :

        psi = np.arctan2(y[i]-y[i-1],x[i]-x[i-1])

        # borne haute du couloir
        lane_x_h.append(x[i] - l*np.sin(psi))
        lane_y_h.append(y[i] + l*np.cos(psi))
        # borne basse du couloir
        lane_x_l.append(x[i] + l*np.sin(psi))
        lane_y_l.append(y[i] - l*np.cos(psi))

    return lane_x_h,lane_y_h,lane_x_l,lane_y_l

def plot_computed_motions(initial_state,states_succession, graphical=True):
    
    #global seq_fin
    global pot_field_trajectory
    global cost_tolerance
    #global states_succession
    global obstacles
    global indice_graph
    
    
    # trajectoire par la méthode des champs de potentiel
    plt.plot(pot_field_trajectory[0],pot_field_trajectory[1])
    
    # obstacles
    for obstacle in obstacles :
        plt.scatter(obstacle.x,obstacle.y,4000)
    
    # "tunnel" dans lequel peuvent évoluer les primitives de mouvement
    #plt.plot(pot_field_trajectory[0], map(lambda x : x + cost_tolerance,pot_field_trajectory[1]))
    #plt.plot(pot_field_trajectory[0], map(lambda x : x - cost_tolerance,pot_field_trajectory[1]))
    

    X = []
    Y = []
            
    new_state = initial_state
    X.append(new_state.x)
    Y.append(new_state.y)
    
    for i in states_succession :
        
        #print new_state.x, new_state.y
        #X.append(new_state.x)
        #Y.append(new_state.y)
    
        for j in i.states :
            X.append(j.x + new_state.x)
            Y.append(j.y + new_state.y)
        new_state = new_state_from_maneuver(new_state,i.q_to,i.id_maneuver)
    
    if graphical :
        plt.plot(X,Y)
        plt.show()
    else :
        plt.plot(X,Y)
        plt.savefig('traj-'+str(indice_graph)+'.jpg')
        indice_graph += 1

"""
Permet de tester le framework de commande pour évitement d'obstacles

"""
def test_framework() :
    
    
    global t
    global obstacles
    global seq_fin
    global motions_types
    global trajectory
    global pot_field_trajectory
    global U_ref
    global X_ref
    global motions_primitives_trajectory
    global time_limit
    global cost_tolerance

    time_limit = 9
    
    print "[*] Détermination des primitives utilisables"
    display_available_primitives()
    
    print ""
    print "     #####################################################################"
    print "     ######################   TEST FRAMEWORK   ###########################"
    print "     #####################################################################"
    print ""
    
    print "[*] Definition des obstacles et positions initiale/finale"    
    initial_state   = Output(0,0,np.radians(45),0,0)
    target_state    = trajectory[0]
    #set_random_obstacles()
    
    print "[*] Generation de la trajectoire par methode des champs de potentiel"
    pot_field_trajectory = set_potential_field_trajectory(initial_state,target_state)
    
    t_1 = time.time()
    
    print "[*] Calcul de la série de primitive de mouvement"
    print "primitives utilieés : "

    plt.plot(pot_field_trajectory[0],pot_field_trajectory[1])

    lane_x_h,lane_y_h,lane_x_l,lane_y_l = get_path_lane(pot_field_trajectory[0],pot_field_trajectory[1],cost_tolerance)
    plt.plot(lane_x_h,lane_y_h)
    plt.plot(lane_x_l,lane_y_l)
    plt.show()
    
    seq_fin, states_succession = approximate(initial_state, pot_field_trajectory)
    
    t_2 = time.time()

    
    print "> Etats résultant des primitives de mouvement précédentes"
    figure("Trajectoire optimale issue des primitives de mouvement")
    plot_computed_motions(initial_state, states_succession)
    
    print "> Temps calcul sequence primitives   : ",(t_2-t_1)
    
    print "[*] Préparation du café et des petits biscuits"
    # préparation du café et des petits biscuits
    
    #print "[*] Suivi de trajectoire par commande prédictive "
    #test_mpc()

    
    
if __name__ == '__main__':

    test_framework()
