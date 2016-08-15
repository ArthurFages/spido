#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sympy as sp

import pylab as pl
import numpy as np
from random import randint
from parameters import *

import mpmath
import time
from scipy.linalg import expm3, expm2, expm

# initial simulation state

"""
Intègre tout les coefficients d'une matrice
"""
def integrate_m(M,len_lin,len_col,t0,tf,dt) :
    x = sp.Symbol("x")
    result = sp.zeros(len_lin,len_col)
    for i in range(0,len_lin) :
        for j in range(0,len_col) :
            result[i,j] = fast_integration(M[i,j],t0,tf,dt)
    return result

def fast_integration(f,t0,tf,dt) :
    x = sp.Symbol("x")
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
    result = sp.zeros(lenA)
    for i in range(0,k) :
        result += (1/fact(i))*(A**i)
    return result

def get_Ad(current_state,T) :
            
    return expm(get_model_a_matrix_lin(current_state.copy())*T)

def get_Bd(current_state, T) :

    global exponential_factor
    global integration_factor

    A   = get_model_a_matrix_lin(current_state.copy())
    b   = get_model_b_matrix()
    B   = np.matrix([[b[0],b[1]],[b[4],b[5]],[0,0],[0,0],[0,0]])

    x   = sp.Symbol("x")

    t1 = time.time()
    exp_at = fast_expm(sp.Matrix(A)*x,5,exponential_factor)
    t2 = time.time()

    #print "Temps de calcul de l'exponentielle : ",(t2-t1)

    to_integrate = exp_at*B

    #print "expt at : ",exp_at
    #print "to integrate : ",to_integrate
    
    t1 = time.time()
    Bd = integrate_m(to_integrate,5,2,0,T,T/integration_factor)
    t2 = time.time()

    #print "Temps de calcul de l'integration : ",(t2-t1)

    return Bd


def set_random_obstacles() :
    
    global obstacles
    
    # détermination aléatoire des obstacles entre ([15,15],[15,15])
    for i in pl.frange(-10,10,3) :
        for j in pl.frange(-10,10,3) :
            rand = randint(0,9)
            if rand>7 :
                obstacles.append(Obstacle(i,j))
                
def get_model_a_matrix() :
    
    global Cf
    global Cr
    global a
    global b
    global M
    global Iz
    
    global Vx
 
    a11 = 2*(Cf+Cr)/(M*Vx)
    a12 = 2*(a*Cf-b*Cr)/(M*Vx)-Vx
    a21 = 2*(a*Cf-b*Cr)/(Vx*Iz)
    a22 = 2*(a*a*Cf+b*b*Cr)/(Vx*Iz)
    
    return [a11, a12, a21, a22]
    
def get_model_a_matrix_lin(current_state) :
    
    a = get_model_a_matrix()
    
    return np.array([[a[0],a[1],0,0,0],
            [a[2],a[3],0,0,0],
            [0,1,0,0,0],
            [-np.sin(current_state.psi),0,-current_state.Vx*np.sin(current_state.psi)-current_state.Vy*np.cos(current_state.psi),0,0],
            [np.cos(current_state.psi),0,current_state.Vx*np.cos(current_state.psi)-current_state.Vy*np.sin(current_state.psi),0,0]])
    
def get_model_a2_matrix_lin(current_state) :

    #print "current state : ",current_state.to_string_v()
    
    global Cf
    global Cr
    global a
    global b
    global M
    global Iz

    # état du spido à un instant t
    Vpsi    = float(current_state.Vpsi)
    Vx      = float(current_state.Vx) # supposée constante
    Vy      = float(current_state.Vy)
    
    psi     = float(current_state.psi)
    x       = float(current_state.x)
    y       = float(current_state.y)

    # gestion des cas particuliers
    if Vx == 0 :
        Vx = 0.00000001

    # coefficients inhérents aux propriétés mécaniques du spido
    a11 = 2*(Cf+Cr)/(M)
    a12 = 2*(a*Cf-b*Cr)/(M)
    a21 = 2*(a*Cf-b*Cr)/(Iz)
    a22 = 2*(a*a*Cf+b*b*Cr)/(Iz)
    
    # calcul du linéarisé
    return np.matrix([
            [a22/Vx,a21*Vy*(-1/Vx**2)+a22*Vpsi*(-1/Vx**2),a21/Vx,0,0,0],
            [0,0,0,0,0,0],
            [a12/Vx-Vx, a11*Vy*(-1/Vx**2)+a12*Vpsi*(-1/Vx**2)-Vpsi,a11/Vx,0,0,0],
            [1,0,0,0,0,0],
            [0,np.cos(psi),-np.sin(psi),-Vx*np.sin(psi)-Vy*np.cos(psi),0,0],
            [0,np.sin(psi),np.cos(psi),Vx*np.cos(psi)-Vy*np.sin(psi),0,0]
        ])
    
    
def get_model_b_matrix() :
    
    global Cf
    global Cr
    global a
    global b
    global d
    global M
    global Iz
    
    global SCxf
    global SCxr
    
    b11 = (SCxf-2*Cf)/M
    b12 = (SCxr-2*Cr)/M
    b13 = 0
    b14 = 0
    b21 = a*(SCxf-2*Cf)/Iz
    b22 = b*(-SCxr+2*Cr)/Iz
    b23 = d/Iz
    b24 = 2/Iz
    
    return [b11, b12, b13, b14, b21, b22, b23, b24]

def get_H_matrix(current_state) :

    global horizon_iteration
    global T_discretisation

    A = np.matrix(get_Ad(current_state, T_discretisation))
    B = np.matrix(get_Bd(current_state, T_discretisation))

    line    = []
    H       = []

    for i in range(1,horizon_iteration+1) :
        for j in range(1,horizon_iteration+1) :
            if i >= j :
                line.append((A**(i-j)).dot(B))
            else :
                line.append(0)
        H.append(line)
    
    return np.matrix(H)

def get_P_matrix(wy,size) :
    return np.matrix(wy*np.eye(size))

def get_D_matrix(size) :
    D = []
    for i in range(0,size) :
        line = []
        for j in range(0,size) :
            if j == i :
                line.append(1)
            elif j == i+1 :
                line.append(-1)
            else :
                line.append(0)

        D.append(line)
    return np.matrix(D)

def get_R_matrix(wu,size) :

    global horizon_iteration

    D = get_D_matrix(horizon_iteration)
    return wu*D.transpose()*D

def get_G_matrix(current_state) :

    global horizon_iteration
    global T_discretisation

    A = get_Ad(current_state, T_discretisation)
    result = []
    for i in range(1,horizon_iteration+1) :
        result.append(A**i)
    return np.matrix(result).transpose()
                
def deriv(syst, t):
    
    global Vx
    global br
    global bf
    
    a = get_model_a_matrix()
    b = get_model_b_matrix()
    
    Vy    = syst[0]
    Vpsi  = syst[1]
    psi   = syst[2]
    X     = syst[3]
    Y     = syst[4]
    
    dVy   = a[0]*Vy + a[1]*Vpsi + b[0]*bf + b[1]*br
    dVpsi = a[2]*Vy + a[3]*Vpsi + b[4]*bf + b[5]*br
    dpsi  = Vpsi
    dX    = Vx*(np.cos(psi)) - Vy*(np.sin(psi))
    dY    = Vx*(np.sin(psi)) + Vy*(np.cos(psi))
    
    return np.array([dVy,dVpsi,dpsi,dX,dY]) 

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def set_offline_model_range() :

    # génération des angles de position du robot autour desquels on va linéariser
    global psi_range
    global Ad_range
    global Bd_range

    global Vx

    global T_discretisation

    print "Offline model linearisation and discretisation"

    pas = np.radians(1)
    for psi in frange(0,np.radians(360),pas) :
        print "[*] psi : ",psi
        psi_range.append(psi)
        Ad_range.append(get_Ad(State(0,Vx,0,psi,0,0,0,0,0),T_discretisation))
        Bd_range.append(get_Bd(State(0,Vx,0,psi,0,0,0,0,0),T_discretisation))




