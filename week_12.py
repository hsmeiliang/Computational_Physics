# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt

def pre_RK4(f, a, h, y0):
    k1 = h*f(a, y0)
    k2 = h*f(a+h/2., y0+k1/2.)
    k3 = h*f(a+h/2., y0+k2/2.)
    k4 = h*f(a+h, y0+k3)
    return a+h, y0+(k1+2*k2+2*k3+k4)/6.
def RK4(f, a, b, N, y0):
    sol = []
    x = []
    sol.append(y0)
    x.append(a)
    h = (b-a)/float(N)
    while a+h < b:
        a, y0 = pre_RK4(f, a, h, y0)
        sol.append(y0)
        x.append(a)
    a, y0 = pre_RK4(f, a, b-a, y0)
    sol.append(y0)
    x.append(a)
    return np.array(x), np.array(sol)

def func9_1(t, y):
    return np.array([y[1], -np.sin(y[0])])

def func9_2(t,y):
    return np.array([y[1], -y[0]])

u0 = np.array([1., 0.])
x0, y0 = RK4(func9_1, 0., 20., 1000, u0)
x1, y1 = RK4(func9_2, 0., 20., 1000, u0)

#plt.plot(x0, y0[:, 0])
#plt.plot(x1, y1[:, 0])
#plt.show()

G = 6.673e-11
M = 1.99e30
a = 2.e8
u0 = np.array([2.e11, 0., 0., 1.e4])
def Gravity(t, r):
    R = (r[0]**2 + r[2]**2)
    return np.array([r[1], -G*M*r[0]/(R**1.5)*(1.-a/R**0.5), r[3], -G*M*r[2]/(R**1.5)*(1.-a/R**0.5)])
T = 60*60*24*365*5
dt = 1000
Nt = int(T/dt)
x, y = RK4(Gravity, 0., T, Nt, u0)
#plt.plot(y[:, 0], y[:, 2])
#plt.show()

I = 1.e-4
m = 0.5
k = 5.
d = 1.e-3
eps = 1.e-2

def Toy(t, Y):
    return np.array([Y[1], -k*Y[0]/m-0.5*eps*Y[2]/m, Y[3], -d*Y[2]/I-0.5*eps*Y[0]/I])

U1 = np.array([0., 0., 2*np.pi, 0.])
x, y = RK4(Toy, 0., 80., 1000, U1)
plt.plot(x, 100*y[:, 0])
plt.plot(x, y[:, 2])
plt.show()































