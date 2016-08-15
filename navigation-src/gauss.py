#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

simulation_size = 1000
m = simulation_size/2
sigma = simulation_size/8
angle_braquage = lambda x: np.radians(30)*np.exp((-1/2)*(((x-m)/sigma)**2))
x = pl.frange(0,simulation_size,1)

plt.figure("30")
plt.plot(x,angle_braquage(x))
plt.show()

angle_braquage = lambda x: np.radians(-30)*np.exp((-1/2)*(((x-m)/sigma)**2))
x = pl.frange(0,simulation_size,1)

plt.figure("-30")
plt.plot(x,angle_braquage(x))
plt.show()
