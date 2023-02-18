# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:06:47 2019

@author: user
"""
import numpy as np
import pylab as plt
import cmath
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy as sci
from scipy import special
import time


def func(x):
    return x**(0.5)
def forwarddiff(f, x, h):
    return (f(x-h)-2*f(x)+f(x+h))/(h**2)
def Cendiff(f, x, h):
    return (2*f(x)-5*f(x+h)+4*f(x+2*h)-f(x+3*h))/(h**2)
def fivediff(f, x, h):
    return (-f(x-2*h)+16.*f(x-h)-40.*f(x)+16*f(x+h)-f(x+2*h))/(12.*h**2)

x = 2
y = -1/4./2**(1.5)
err = []
h = np.array([0.2, 0.1, 0.05, 0.01, 0.001, 1.e-4, 1.e-8, 1.e-10, 1.e-12, 1.e-14, 1.e-16])
result1 = np.abs((forwarddiff(func, x, h)-y)/y)
plt.plot(h, result1, 'sc', label = 'forward', color = 'blue')
result2 = np.abs((Cendiff(func, x, h)-y)/y)
plt.plot(h, result2, 'ob', label = 'Central', color = 'black')
result3 = np.abs((fivediff(func, x, h)-y)/y)
plt.plot(h, result3, 'vr', label = 'fivepoint', color = 'red')
plt.ylabel(r'Relative errors', fontsize = 15)
plt.xlabel(r'h', fontsize = 15)
#plt.ylim(1.e-5, 1e40)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()