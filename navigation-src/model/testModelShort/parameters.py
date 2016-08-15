#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

from simu_classes import *
import numpy as np




# == Paramètres de simulation ==================================================

horizon_iteration   = 1000         # nombre d'itérations à considérer dans la commande prédictive
T_discretisation    = 0.02     

################################
#
#
# ATTENTION : 
# - Ne pas prendre T_discretisation trop grand sinon calcul de Vy instable
# - Ne pas prendre une horizon trop grande, sinon discrétisation instable
# 
################################


# == Paramètres du modèle ======================================================

Cf      = 100 # rigidité de dérive du train avant
Cr      = 100 # rigidité de dérive du train arrière
M       = 900
Iz      = 1700
a       = 1.099
b       = 1.099
d       = 0.8
betaMax = np.radians(40) # max wheel rotation 


SCxf    = 100.1
SCxr    = 100.1


Vx = 4





# == Paramètres de minimisation des fonctions de coût ==========================

reccursion_limit    = 1000
nb_rec              = 0
end_recursion       = False
go_next_trim        = False


# == Discrétisation offline du modèle ============================================
psi_range   = []
Ad_range    = []
Bd_range    = []



# == Paramètres de calculs ======================================================

integration_factor  = 3
exponential_factor = 3



