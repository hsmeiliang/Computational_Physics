# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
import scipy as sci

def Trapez_1(f, a, b, N):
    h = (b-a)/float(N)
    x = np.linspace(a, b, N+1)
    w = np.ones(N+1)*h
    w[0] = w[N] = h/2.
    return np.sum(f(x)*w)
def F1(x):
    return x**4-2.*x+1

print(Trapez_1(F1, 0., 2., 10000))

def Simpson_1(f, a, b, N):
    h = (b-a)/float(N)
    x1 = np.arange(a+h, b, 2*h)
    x2 = np.arange(a+2*h, b, 2*h)
    w1 = np.ones(len(x1))*h*4/3.
    w2 = np.ones(len(x2))*h*2/3.
    return np.sum(f(x1)*w1) + np.sum(f(x2)*w2) + h/3.*(f(a)+f(b))

print(Simpson_1(F1, 0., 2., 10000))


def myGL(f, a, b, N):
    x, w = np.polynomial.legendre.leggauss(N)
    xp = 0.5*(b-a)*x + 0.5*(b-a)
    wp = 0.5*(b-a)*w
    return np.sum(f(xp)*wp)
print(myGL(F1, 0., 2., 3))

thetaD = 428.
rho = 6.022*10**-28
V = 1000.0e-6
kB = 1.3806488e-23

def cv(t):
    def g(x):
        return x**4*np.exp(x)/(np.exp(x)-1.)**2
    return 9*V*rho*kB*(t/thetaD)**3*myGL(g, 0., thetaD/t, 50)

T = np.arange(5, 501, 5.)
CV = []
for i in T:
    CV.append(cv(i))
plt.plot(T/thetaD, CV)
plt.show()


def T(a, n):
    def g(x):
        return (8./(a**n-x**n))**0.5
    return myGL(g, 0.,a, 50)

t = np.linspace(0.1, 1.0, 21)
period1 = []
period2 = []
period3 = []

for i in t:
    period1.append(T(i, 1))
    period2.append(T(i, 2))
    period3.append(T(i, 3))
    

plt.plot(t, period1)
plt.plot(t, period2)
plt.plot(t, period3)
plt.show()

z = 3
x = np.linspace(-5, 5, 100)

def C(u):
    def g(x):
        return np.cos(np.pi/2.*x**2)
    return myGL(g, 0, u, 50)
def S(u):
    def g(x):
        return np.sin(np.pi/2.*x**2)
    return myGL(g, 0, u, 50)
lam = 1.
U = x*np.sqrt(2./lam/z)
intensity = []
for i in U:
    intensity.append(1./8.*((2*C(i)+1.)**2+(2*S(i)+1.)**2))
plt.plot(x, intensity)
plt.show()











