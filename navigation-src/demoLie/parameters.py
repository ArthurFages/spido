#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

from classes import *

import numpy as np

# == Relevés des erreurs =======================================================

erreurs_u = []
erreurs_x = []
erreurs_y = []

# == Paramètres d'état =========================================================

TIME    = 0
t       = 0
X       = 0
Y       = 0
PSI     = 0
VY      = 0
VX      = 0
VPSI    = 0

# == Paramètres du signal de référence =========================================

k_w = 0.01
W   = []
W_reel  = []
x_t     = []
y_t     = []
x_t_reel= []
y_t_reel= []
psi_t   = []
x_dev_t = []
y_dev_t = []
U_ref   = []
X_ref   = []

# == Paramètres de la trajectoire ==============================================

trajectory  = [Output(40,0,0,0,0)] #[Output(40,5,0,0,0)] # cible
i_traj      = 0
target      = trajectory[i_traj]
tolerance   = 0.9
states_ref  = []

# == Paramètres de simulation ==================================================



#horizon_iteration   = 10       # nombre d'itérations à considérer dans la commande prédictive
#T_discretisation    = 0.1      # Pas de discrétisation


# == Paramètres du modèle ======================================================

Cf      = 1500#7500#12218.5 # rigidité de dérive du train avant
Cr      = 1500#7500#8313.0 # rigidité de dérive du train arrière
M       = 900#1750
Iz      = 1300#3000#1500
a       = 1.1#0.85#1.099 #1.1
b       = 1.1#0.85#1.099 #1.6
d       = 0.75#0.35#0.8
betaMax = np.radians(40) # max wheel rotation 


#SCxf    = 0
#SCxr    = 0


Vx = 1


# == Paramètres de stockage d'information ======================================

betar_tab   = [0]
betaf_tab   = [0]

x           = [0]
y           = [0]
psi_tab     = [0]
Vy_tab      = [0]
Vpsi_tab    = [0] 
#time        = [0]    

X_tot   = []
Y_tot   = []
psi_tot = []


            
# == Gestion des primitives ======================================================            
            
obstacles   = [Obstacle(25,5)]
                
risk_zone = 1 # rayon autour de l'obstacle représentant une zone à risque
                
psi_pot = []

Kr      = 01     # écart avec la trajectoire de référence
K0      = 1     # considération de l'obstacle
epsilon = 0.2 # marge par rapport à l'obstacle

Kmr         = 1 # écart avec la trajectoire de référence
Kmp         = 1 # considération de l'angle de braquage
alpha_max   = np.radians(40)

pos_trajectory_ref = [] # trajectoire de référence
pos_trajectory_eff = [] # trajectoire effective <=> réelle

trim_man_trajectory = [] # ensemble des maneuvers et trim à réaliser par le robot
pot_field_trajectory = [] # trajectoire selon la méthode des champs de potentiel

motions_types   = []
seq             = []
seq_fin         = []

# ==  Paramètres champs de potentiel ===========================================

k   = 1 # pas de considération des angles
k2  = 0.05 # coefficient proportionnel d'évolution de la position

# == Paramètres de calculs ======================================================

integration_factor  = 3
exponential_factor = 3
thau_rk = 0.05 # à faire dépendre de T_discretisation

# == Paramètres de minimisation des fonctions de coût ==========================

reccursion_limit    = 1000
nb_rec              = 0
end_recursion       = False
go_next_trim        = False

# == Paramètres de lissage de la trajectoire par primitive de mvt ==============

time_limit          = 60   # secondes
cost_tolerance      = 1     # largeur du "tunnel" autour de la trajectoire de référence 
                            # dans lequel peuvent se situer les primitives de mouvement
obstacle_tolerance  = 2     # définit la "zone d'effet" d'un obstacle

















# == Chargement des primitives ===================================








# TODO : stocker à terme tout les paramètres a et L dans un ou deux tableaux
trims       = [ Trim(0,Vx,0,0,"Tout droit",0),
                Trim(1,Vx,0.01,-0.3,"Virage droite",0),
                Trim(2,Vx,-0.01,0.3,"Virage gauche",0)]
maneuvers   = [Maneuver(0,0,0,0,0.5,0,0,"tout droit")]#,
                #Maneuver(1,0,0,np.radians(45),0.5,0,0,"virage gauche 20 long"),
                #Maneuver(2,0,0,-np.radians(45),0.5,0,0,"virage droite 20 long"),
                #Maneuver(3,0,0,np.radians(25),0.5,0,0,"virage gauche 10 court"),
                #Maneuver(4,0,0,-np.radians(25),0.5,0,0,"virage droite 10 court")]#,

# chargement de la trajectoire canonioque de référence
#set_maneuver_0(horizon_iteration,T_discretisation)

# chargement virages à gauche
#primitive_from_file("virage_gauche",1)
#primitive_from_file("virage_gauche_court_10",3)

# chargement virages à droite
#primitive_from_file("virage_droite",2)
#primitive_from_file("virage_droite_court_10",4)

# chargement droite inclinée
#primitive_from_file("virage_droite",5)


# vérification des maneuvres enregistrées

#print "[*] Vérification des maneuvres enregistrées "
#for maneuver in maneuvers :
#    print "> ",maneuver.to_string()
            
