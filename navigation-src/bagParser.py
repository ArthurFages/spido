#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    
    print "# Récupération des données dans le fichier bag"
    
    X = []
    Y = []
    Z = []
    

    with open('2experience.txt') as bag:
        for line in bag:
            #print line
            if(line[0]!='%') :
                coordonates = line.split(',')
                #print coordonates
                X.append(coordonates[4])
                Y.append(coordonates[5])
                Z.append(coordonates[6])
        
    bag.close()
    
    print "# affichage des données"
    
    print "len X",len(X)
    print "len Y",len(Y)
    print "len Z",len(Z)
    
    fig = plt.figure("prout")
    
    
    
    # champs de potentiel
    ax = Axes3D(fig)
    
    #X_map = np.arange(-100, 100, 5)
    #Y_map = np.arange(-100, 100, 5)
    #X, Y = np.meshgrid(X, Y)
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    
    X = map(float,X)
    Y = map(float,Y)
    Z = map(float,Z)
    
    X = X[0:len(X)-200]
    Y = Y[0:len(Y)-200]
    Z = Z[0:len(Z)-200]
    
    #plt.show()
    #plt.mplot3D(X,Y,Z)
    ax.plot_wireframe(X,Y,Z)
    plt.show()
