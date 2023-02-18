# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:36:54 2019

@author: user
"""

import numpy as np
import pylab as plt

def pre_euler(f, x0, y0, h):
    return y0 + h*f(x0, y0)

def euler(f, x0, xf, N, y0):
    h = (xf - x0)/float(N)
    sol = []
    x = []
    sol.append(y0)
    x.append(x0)
    while x0+h < xf:
        y0 = pre_euler(f, x0, y0, h)
        sol.append(y0)
        x0 = x0 + h
        x.append(x0)
    sol.append(pre_euler(f, x0, sol[-1], xf-x0))
    x.append(xf)
    return np.array(x), np.array(sol)

def func(x, y):
    return -1*(y/x - 1/x**2 + y**2)

x, y = euler(func, 1, 2, 100, -1)
#plt.plot(x, y)
#plt.savefig('1.png')

def pre_imp_euler(f, x0, y0, h):
    yt = y0 + h*f(x0, y0)
    return x0 + h, y0 + h/2.*(f(x0, y0)+f(x0+h, yt))

def imp_euler(f, x0, xf, N, y0):
    h = (xf - x0)/float(N)
    sol = []
    x = []
    sol.append(y0)
    x.append(x0)
    while x0+h < xf:
        x0, y0 = pre_imp_euler(f, x0, y0, h)
        sol.append(y0)
        x.append(x0)
    x0, y0 = pre_imp_euler(f, x0, sol[-1], xf-x0)
    sol.append(y0)
    x.append(x0)
    return np.array(x), np.array(sol)

x, y = imp_euler(func, 1, 2, 100, -1)
#plt.plot(x, y)
#plt.savefig('2.png')

def func5(x, y):
    return -y**3+np.sin(x)
x, y = imp_euler(func5, 0., 10., 100, np.array([0,0]))
#plt.plot(x, y)
#plt.savefig('3.png')


def func7(t, z):
    return np.array([1.*z[0]-0.5*z[0]*z[1], 0.5*z[0]*z[1]-2.*z[1]])

z = np.array([2, 2])
x, y = imp_euler(func7, 0, 20, 100, z)
#plt.plot(x, y[:, 0])
#plt.plot(x, y[:, 1])
#plt.savefig('4.png')

def func8(t, y):
    return np.array([y[0]*y[1]-y[0], y[1]-y[0]*y[1]+np.sin(t)**2])
z = np.array([1.0, 1.0])
x, y = imp_euler(func8, 0, 10, 100, z)
plt.plot(x, y[:, 0])
plt.plot(x, y[:, 1])
plt.show()


    









