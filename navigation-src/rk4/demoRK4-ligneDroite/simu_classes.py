#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# classes for simulation

# Command : angles indiqués en radians
class Command :
    def __init__(self, betaf, betar) :
        
        global betaMax
        
        self.betaf = betaf
        self.betar = betar
    
    def to_string(self) :
        return str(self.betaf)+" : "+str(self.betar)

# Output <=> état du système : angles indiqués en radians !! => convertir avant
class Output :
    def __init__(self, x, y, psi, Vy, Vpsi, t=0) : 
    
        #print "[*] New position : x "+str(x)+" - y "+str(y)+" - psi "+str(psi)+" - Vy "+str(Vy)+" - Vpsi "+str(Vpsi) 
    
        self.x      = x
        self.y      = y
        self.psi    = psi
        self.Vy     = Vy
        self.Vpsi   = Vpsi
        self.t      = t
    
    def to_string(self) :
        return "x "+str(self.x)+" - y "+str(self.y)+" - psi "+str(self.psi)+" - Vy "+str(self.Vy)+" - Vpsi "+str(self.Vpsi) 
    
class State(Output) :
    def __init__(self, Vpsi, Vx, Vy, psi, x, y, beta_f, beta_r, t=None) :
        Output.__init__(self, x, y, psi, Vy, Vpsi)
        self.Vx     = Vx
        self.beta_f = beta_f
        self.beta_r = beta_r
        self.t      = t
        
    def to_string(self) : 
        return str(self.t)+","+str(self.beta_f)+","+str(self.beta_r)+","+str(self.Vpsi)+","+str(self.Vx)+","+str(self.Vy)+","+str(self.psi)+","+str(self.x)+","+str(self.y) 
    
    def to_string_v(self) :
        return "t: "+str(self.t)+", bf "+str(self.beta_f)+", br "+str(self.beta_r)+", Vpsi "+str(self.Vpsi)+", Vx "+str(self.Vx)+", Vy "+str(self.Vy)+", psi "+str(self.psi)+", x  "+str(self.x)+", y "+str(self.y) 
    
    
    def copy(self) :
        return State(self.Vpsi, self.Vx, self.Vy, self.psi, self.x, self.y, self.beta_f, self.beta_r, self.t)
        
    def __sub__(self, autre_state) :
        return State(self.Vpsi-autre_state.Vpsi, self.Vx-autre_state.Vx, self.Vy-autre_state.Vy, self.psi-autre_state.psi, self.x-autre_state.x, self.y-autre_state.y, self.beta_f-autre_state.beta_f, self.beta_r-autre_state.beta_r, self.t-autre_state.t)
        
    def __add__(self, autre_state) :
        return State(self.Vpsi+autre_state.Vpsi, self.Vx+autre_state.Vx, self.Vy+autre_state.Vy, self.psi+autre_state.psi, self.x+autre_state.x, self.y+autre_state.y, self.beta_f+autre_state.beta_f, self.beta_r+autre_state.beta_r, self.t+autre_state.t)
        
        
# Obstacles : stuff to avoid
# hypothèse : les obstacles sont supposés ponctuels (relevés du lidar)
# size = rayon
class Obstacle :
    def __init__(self, x,y) :
        self.x = x
        self.y = y
    def to_string(self) :
        return "Obstacle - x "+str(self.x)+" - y "+str(self.y)
        
class Trim :
    # thau : time to realize the trim
    def __init__(self, id_trim, Vx, Vy, Vpsi, description, thau=0, beta_f=0, beta_r=0) :
        self.id_trim        = id_trim
        self.Vx             = Vx
        self.Vy             = Vy
        self.Vpsi           = Vpsi
        self.description    = description
        self.thau           = thau
        self.beta_f         = beta_f
        self.beta_r         = beta_r
        
    def to_string(self) :
        return "Trim ",self.description," ",self.id_trim," : Vx "+str(self.Vx)+" - Vy "+str(self.Vy)+" - Vpsi "+str(self.Vpsi)+" - thau : ",self.thau
        
    def copy(self) :
        return Trim(self.id_trim, self.Vx, self.Vy, self.Vpsi, self.description, self.thau)
        
class Maneuver :
    def __init__(self, id_maneuver, delta_x, delta_y, delta_psi, thau, q_from, q_to,description, states=[]) :
        self.id_maneuver= id_maneuver
        self.delta_x    = delta_x
        self.delta_y    = delta_y
        self.delta_psi  = delta_psi
        self.thau       = thau
        self.q_from     = q_from # initial trim
        self.q_to       = q_to   # final trim
        self.description= description
        self.states     = states
    
    def to_string(self) :
        return "Maneuver ",self.description," ",self.id_maneuver," Dx "+str(self.delta_x)+" - Dy "+str(self.delta_y)+" - Dpsi "+str(self.delta_psi)+" - Thau : "+str(self.thau)
        
    def copy(self) :
        new_states = []
        for state in self.states :
            new_states.append(state.copy())
        return Maneuver(self.id_maneuver, self.delta_x, self.delta_y, self.delta_psi, self.thau, self.q_from, self.q_to,self.description,new_states)
        
class Point :
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_string(self):
        print "(",self.x,";",self.y,")"
      
class Node :
    def __init__(self, prec, heuristic, cost, theta, point, man, i_man, isTarget=False, isRoot=False) :
        self.prec       = prec
        self.h          = heuristic
        self.c          = cost
        self.theta      = theta
        self.pt         = point
        self.man        = man # maneuvre pour obtenir ce Node (orientée)
        self.i_man      = i_man # ref de la maneuvre (non orientée)
        self.isTarget   = isTarget
        self.isRoot     = isRoot
    def to_string(self) :
        return "prec : ",self.prec," is root : ",self.isRoot
# =========================================================================================

