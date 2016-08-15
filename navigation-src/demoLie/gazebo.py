#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import rospy
import time

from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64, Float32, Int32, String
from spido_pure_interface.msg import cmd_car

from parameters import *
from MPC import *
from primitives import *

# == Ressources partagées pour l'exécution parrallèle ============================

STATE = None # état du robot à un instant t
BETAF = 0
BETAR = 0

T0 = 0          # origine des temps
T = 0           # temps actuel

# repère de départ
X0 = 0
Y0 = 0
PSI0 = 122

# ====================================================================================

# == Drapeaux pour la gestion de l'execution =========================================

STOP_PROCESS			= False
STOP_PUBLISH_COMMAND	= False

# ====================================================================================



def get_topic_values() :
    global STATE, BETAR, BETAF, T
    return STATE, BETAR, BETAF, T

"""
Enregistrement du temps simulé
"""
def new_time(data):
    global T
    sec     = data.clock.secs   # secondes
    nsec    = data.clock.nsecs  # nanosecondes
    T = sec+nsec*(1e-9)            # conversion en secondes

"""
Enregistrement de l'état du SPIDO à un instant donné
"""
def new_position(data):
    
    global T
    global T0
    
    global PSI0
    global X0
    global Y0

    global STATE
    global BETAR 
    global BETAF
    
    dt = 0
    
    # enregistrement de quand on a fait la mesure
    TIME = T - T0

    #print "["+str(TIME)+"]"
    
    # récupération du torseur position
    X   = data.pose.pose.position.x
    Y   = data.pose.pose.position.y
    sin = data.pose.pose.orientation.z
    cos = data.pose.pose.orientation.w
    if sin < 0 :
        PSI = -2*np.arccos(cos)
    else :
        PSI = 2*np.arccos(cos)
        
    if PSI0 == 122 :
        # initialisation du repère de départ =/= repère global
        PSI0   = PSI
        X0     = X
        Y0     = Y
        
    # on considère les coordonées dans le repère de départ
    delta_x = np.sign(X)*(abs(X) - abs(X0)) 
    delta_y = np.sign(Y)*(abs(Y) - abs(Y0))
    X = delta_x*np.cos(PSI0) + delta_y*np.cos(np.radians(90) - PSI0)
    Y = delta_y*np.cos(PSI0) - delta_x*np.cos(np.radians(90) - PSI0)
    PSI = PSI - PSI0
    
    # calcul des vitesses dans le repère global
    VX_abs = data.twist.twist.linear.x
    VY_abs = data.twist.twist.linear.y
    
    # calcul des vitesses dans le repère lié au robot
    VPSI = data.twist.twist.angular.z
    VX = VX_abs*np.cos(PSI) + VY_abs*np.cos(np.radians(90)-PSI)
    VY = VY_abs*np.cos(PSI) - VX_abs*np.cos(np.radians(90)-PSI)

    STATE = State(VPSI, VX, VY, PSI, X, Y, BETAF, BETAR, TIME)
    #states.append(STATE)

def set_rospy_listener() :

    global T0

    # réception de l'état réel du robot (position+temps)
    rospy.Subscriber("odom", Odometry, new_position)
    rospy.Subscriber("clock", Clock, new_time)
    T0 = T
    time.sleep(2)

def gazebo_physic_management() :

	rospy.wait_for_service('gazebo/pause_physics')
	pause_physics = rospy.ServiceProxy("gazebo/pause_physics", Empty)

	rospy.wait_for_service('gazebo/unpause_physics')
	unpause_physics = rospy.ServiceProxy("gazebo/unpause_physics", Empty)

	return pause_physics, unpause_physics

def initiate_ref_signal(ref_starting_state, horizon_iteration, extended=False) :

    global maneuvers

    # détermination de la trajectoire de référence
    print "[*] Définition d'une suite de primitives de mouvement"
    #seq_fin = [maneuvers[0].copy(),maneuvers[1].copy(),maneuvers[2].copy(),maneuvers[3].copy()]
    #seq_fin = [maneuvers[0],maneuvers[2],maneuvers[2],maneuvers[0],maneuvers[0],maneuvers[0],maneuvers[0]]
    for i in range(0,len(maneuvers)-3) :
        seq_fin.append(maneuvers[i])

    print "[*] Définition des trajectoires de référence"
    states_ref = set_state_ref2(ref_starting_state,seq_fin)

    print "len state ref : ",len(states_ref)

    #display_states(states_ref)
    
    Xref    = []
    Yref    = []
    PSIref  = []
    VXref   = []
    VYref   = []
    VPSIref = []
    BFref   = []
    BRref   = []
    for i in range(0,len(states_ref)) :
        Xref.append(states_ref[i].x)
        Yref.append(states_ref[i].y)
        PSIref.append(states_ref[i].psi)
        VXref.append(states_ref[i].Vx)
        VYref.append(states_ref[i].Vy)
        VPSIref.append(states_ref[i].Vpsi)
        BFref.append(states_ref[i].beta_f)
        BRref.append(states_ref[i].beta_r)


    #plt.figure("Trajectoire de référence")
    #plt.plot(Xref,Yref)
    #plt.figure("Psi référence")
    #plt.plot(range(0,len(PSIref)),PSIref)
    #plt.figure("Vx référence")
    #plt.plot(range(0,len(VXref)),VXref)
    #plt.figure("Vy référence")
    #plt.plot(range(0,len(VYref)),VYref)
    #plt.figure("VPsi référence")
    #plt.plot(range(0,len(VPSIref)),VPSIref)
    #plt.show()

    # définition du chemin sur une horizon
    u_ref = get_u_ref(ref_starting_state, states_ref, horizon_iteration)
    if extended :
        x_ref = get_x_ref_extended(ref_starting_state, states_ref, horizon_iteration)
        t_ref = get_t_ref_extended(ref_starting_state, states_ref, horizon_iteration)
    else :
        x_ref = get_x_ref(ref_starting_state, states_ref, horizon_iteration)
        t_ref = get_t_ref(ref_starting_state, states_ref, horizon_iteration)

    return u_ref, x_ref, t_ref, Xref, Yref


def test_gazebo() :

    global STOP_PROCESS
    global STOP_PUBLISH_COMMAND
    
    global betaMax
    global Vx
    
    # temps de prédiction
    horizon_iteration 	= 10
    T_discretisation	= 0.2

    # temps de simulation
    T_discretisation_simulation		= 0.05
    horizon_iteration_simulation 	= 4000
    f_simulation                    = 100


    print "### GAZEBO SIMULATION ###"


    

    
    rospy.init_node('trajectory_following_MPC')
    
    # expression de la commande 
    print "[*] Definition des publishers"
    cmd_safe_publisher  = rospy.Publisher('cmd_car_safe',cmd_car,queue_size=10)
    cmd_publisher       = rospy.Publisher('cmd_car',cmd_car,queue_size=10)
    state_publisher     = rospy.Publisher('state',Int32,queue_size=10)
    err_path_publisher	= rospy.Publisher('err_path',String,queue_size=10)
    ref_path_publisher  = rospy.Publisher('ref_path',String,queue_size=10)
    eff_path_publisher  = rospy.Publisher('eff_path',String,queue_size=10)

    front_r_steer_pub   = rospy.Publisher('spido/front_right_steering/command',Float64,queue_size=10)
    front_l_steer_pub   = rospy.Publisher('spido/front_left_steering/command',Float64,queue_size=10)
    rear_r_steer_pub    = rospy.Publisher('spido/rear_right_steering/command',Float64,queue_size=10)
    rear_l_steer_pub    = rospy.Publisher('spido/rear_left_steering/command',Float64,queue_size=10)

    # listeners
    print "[*] Definition des listeners"
    set_rospy_listener()
    
    # Command
    cmd = cmd_car()

    # taux déchatillonnage avec ROS
    r = rospy.Rate(f_simulation) 
    #r = rospy.Rate(100) # 1 mega hertz dans ta face

    # commande de la vitesse du SPIDO
    #cmd.linear_speed = Vx
    #cmd.steering_angle = 0
    #time.sleep(1)

    #cmd_publisher.publish(cmd)
    #cmd_safe_publisher.publish(cmd)
    #state_publisher.publish(1)
    #time.sleep(1)

    print "\n[*] Chargement primitive utilisée pour le test"
    set_primitives(T_discretisation_simulation)
    set_maneuver_0(int(600/5),T_discretisation_simulation)

    # gestion de la physique de Gazebo
    #pause_physics, unpause_physics = gazebo_physic_management()
    
    # trajectoire effective du SPIDO sous gazebo
    X_results       = [0]
    Y_results       = [0]
    states_results  = []
    times_results   = []
    uf_results		= []
    ur_results		= []
    error_X			= []
    error_Y			= []
    
    # définition du signal de référence (trajectoire à suivre)
    print "[*] Définition des trajectoires de référence"
    STATE, BETAR, BETAF, T = get_topic_values()
    current_state 	= STATE #State(0,Vx,0,0,0,0,0,0,0)
    print "> initial state : ",STATE.to_string_v()
    current_command = Command(BETAF, BETAR)
    print "T : ",type(T)
    u_ref, x_ref, t_ref, Xref, Yref = initiate_ref_signal(current_state, horizon_iteration_simulation)

    #print "x_ref : ",x_ref

    last_t_ref = [-122,-122,-122]

    t1 = time.time()
    
    print "[*] Suivi de ces trajectoires par MPC sous gazebo"
    for i in range(0,horizon_iteration_simulation) :



        if i >= 0 :
            #print "> initial Vx command : ",Vx

            cmd.linear_speed = Vx
            #cmd.steering_angle = 0
            #time.sleep(1)
            cmd_publisher.publish(cmd)
            cmd_safe_publisher.publish(cmd)
            #state_publisher.publish(1)



        
        if STOP_PROCESS :
            print "stop process"
            break

        #print X_results[-1]," / ",x_ref[-1][4]
        #print Y_results[-1]," / ",x_ref[-1][5]

        if ((X_results[-1]**2 + Y_results[-1]**2) - (x_ref[-1][4]**2 + x_ref[-1][5]**2))**2 < 1 :
            print "on est arrivé"
            break

        # relevé position
        current_state, BETAR, BETAF, T = get_topic_values()
        X_results.append(current_state.x)
        Y_results.append(current_state.y)
        states_results.append(current_state)

        # définition des signaux de référence pour la position courante
        x_ref_k, t_ref_k = get_xt_ref_k(current_state,T_discretisation_simulation,x_ref, t_ref)

        #delta_ref   = rospy.Publisher('delta_ref',String,queue_size=10)
    	#delta_ref.publish("Dx = ref : "+str(x_ref_k[1][4])+" / eff: "+str(X_results[-1])+" - Dy = "+str(x_ref_k[1][5])+"/"+str(Y_results[-1])+" - Dt = "+str(t_ref_k[1])+"/"+str(current_state.t))
    	#delta_ref.publish(str(x_ref_k))
        # publication des positions effectives et de référence
        err_path_publisher.publish("err : ("+str(x_ref_k[1][4]-X_results[-1])+","+str(x_ref_k[1][5]-Y_results[-1])+")")
        ref_path_publisher.publish("ref : ( x = "+str(x_ref_k[1][4])+", y = "+str(x_ref_k[1][5])+", psi = "+str(x_ref_k[1][3])+" Vx = "+str(x_ref_k[1][1])+", Vy = "+str(x_ref_k[1][2])+", Vpsi = "+str(x_ref_k[1][0])+" )")
        eff_path_publisher.publish("( x = "+str(current_state.x)+"/"+str(x_ref_k[1][4])+", y = "+str(current_state.y)+"/"+str(x_ref_k[1][5])+", psi = "+str(current_state.psi)+"/"+str(x_ref_k[1][3])+" Vx = "+str(current_state.Vx)+"/"+str(x_ref_k[1][1])+", Vy = "+str(current_state.Vy)+"/"+str(x_ref_k[1][2])+", Vpsi = "+str(current_state.Vpsi)+"/"+str(x_ref_k[1][0])+" )")
        
        # calcul des erreurs
        error_X.append((X_results[-1]-x_ref_k[1][4])**2)
        error_Y.append((Y_results[-1]-x_ref_k[1][5])**2)


        # cas de la fin du signal de référence
        #if t_ref_k[1] < last_t_ref[1] :
        #    print "time bordel"
        #    break
        #last_t_ref = t_ref_k

        # génération de la commande
        u, duration = get_commands_on_horizon_Lie(  T_discretisation,
                                                    horizon_iteration,
                                                    current_state,
                                                    current_command,
                                                    u_ref, 
                                                    x_ref_k, 
                                                    t_ref_k, 
                                                    Xref, 
                                                    Yref)

        # contrôle des dépassements des angles possibles
        #if u[0][0] > betaMax :
        #    u[0][0] = betaMax
        #if u[0][0] < -betaMax :
        #    u[0][0] = -betaMax
        #if u[0][1] > betaMax :
        #    u[0][1] = betaMax
        #if u[0][1] < -betaMax :
        #    u[0][1] = -betaMax

        print "[ ",i,"/",horizon_iteration_simulation,"] > Commande : (",u[0][0],",",u[0][1],") - Vx : ",current_state.Vx
        print current_state.to_string_v()

        uf_results.append(u[0][0])
        ur_results.append(u[0][1])
        current_command = Command(u[0][0],u[0][1])
        times_results.append(duration)

        # envoie de la commande au spido
        #front_r_steer_pub.publish(u[0][0])
        #front_l_steer_pub.publish(u[0][0])
        #rear_r_steer_pub.publish(u[0][1])
        #rear_l_steer_pub.publish(u[0][1])

        cmd.linear_speed = Vx
    	cmd.steering_angle_front = u[0][0]
        cmd.steering_angle_rear = u[0][1]
    	cmd_publisher.publish(cmd)
    	cmd_safe_publisher.publish(cmd)
    	state_publisher.publish(1)

        # temporisation        
        r.sleep()

    t2 = time.time()

    print "> Temps de la simulation : ",(t2-t1)

    # arrêt du robot
    cmd.linear_speed = 0
    cmd.steering_angle_front = 0
    cmd.steering_angle_rear = 0
    cmd_publisher.publish(cmd)
    cmd_safe_publisher.publish(cmd)
    state_publisher.publish(1)
    
    # affichage des résultats
    print "\n[*] Affichage des trajectoires"
    # trajectoire de référence
    plt.figure("Trajectoires")
    plt.plot(Xref, Yref, label="$Ref$")
    # affichage de la trajectoire effectuée par le SPIDO
    plt.plot(X_results, Y_results, label="$Eff$")
    plt.legend()

    # affichage de l'évolution de la commande
    plt.figure("Evolution de la commande")
    plt.plot(range(0,len(uf_results)),uf_results, label="$u_{f}$")
    plt.plot(range(0,len(ur_results)),ur_results, label="$u_{r}$")
    plt.legend()

    # affichage des erreurs
    plt.figure("Erreur selon x et y")
    plt.plot(range(0,len(error_X)),error_X, label="$err_{x}$")
    plt.plot(range(0,len(error_Y)),error_Y, label="$err_{y}$")
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
	test_gazebo()