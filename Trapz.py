# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:07:02 2019

@author: test
"""

import numpy as np
import pylab as plt
import cmath
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy as sci
from scipy import special
from scipy import integrate
import time

def func(x):
    return x**4-2*x+1

def N_and_2N_method(func, mysum, epsilon):
    N = 100
    x = np.linspace(0., 2., N)
    y = func(x)
    s0 = mysum(y,x)
    x = np.linspace(0., 2., 2*N)
    y = func(x)
    s1 = mysum(y,x)
    while abs(s0 - s1) > epsilon :
        N*=2
        s0 = s1
        x = np.linspace(0., 2., 2*N)
        y = func(x)
        s1 = mysum(y, x)
    print(mysum(y, x))
    
epsilon = 1.e-10
N_and_2N_method(func, sci.integrate.trapz, epsilon)
