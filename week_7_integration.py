# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


###problem2
def Trapez(f, a, b, N):
    h = (b-a)/float(N)
    s = (f(a)+f(b))/2
    for i in range(1, N):
        s = s+f(a+i*h)
    return h*s

def func(x):
    return np.exp(-x*x)

print(Trapez(func, 0., 2., 100))
x = np.linspace(0., 2., 100)
y = func(x)
print( sci.integrate.trapz(y,x) )

def Trapez_fast(f, a, b, N):
    h = (b-a)/float(N)
    s = np.linspace(a, b, N+1)
    return h*(np.sum(f(s))-0.5*(f(a)+f(b)))
print(Trapez_fast(func, 0., 2., 100))

def Simpson(f, a, b, N):
    h = (b-a)/float(N)
    s = f(a)+f(b)
    for i in range(1, N, 2):
        s = s + 4.*f(a+i*h)
    for i in range(2, N, 2):
        s = s + 2.*f(a+i*h)
    return h/3.*s
print(Simpson(func, 0., 2., 100))
x = np.linspace(0., 2., 101)
y = func(x)
print(sci.integrate.simps(y,x))

def Simpson_fast(f, a, b, N):
    h = (b-a)/float(N)
    x1 = np.arange(a+h, b, 2*h)
    x2 = np.arange(a+2.*h, b, 2*h)
    return h/3.*(f(a)+f(b)+4.*np.sum(f(x1))+2.*np.sum(f(x2)))
print(Simpson_fast(func, 0., 2., 100))

def func2(x):
    def g(s):
        return np.exp(-s**2)
    return Simpson_fast(g, 0., x, 100)

y = []
x = np.arange(0.1, 3., 0.1)
for i in x:
    y.append(func2(i))
plt.plot(x, y)
plt.show()



def bessel(m, x):
    def integrand(s):
        return np.cos(m*s-x*np.sin(s))
    return 1./np.pi*Simpson_fast(integrand, 0., np.pi, 1000)

lam = 500.e-9
k = 2.*np.pi/lam
x = np.linspace(-1.e-6, 1.e-6, 101)
y = np.linspace(-1.e-6, 1.e-6, 101)
cir = []
for i in x:
    for j in y:
        r = (i**2+j**2)**0.5
        cir.append((bessel(1, k*r)/(k*r))**2)
cir = np.reshape(cir, (101, 101))
plt.imshow(cir, vmax = 0.005)
plt.gray()






















