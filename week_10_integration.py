import numpy as np
import scipy.special
import math
import pylab as plt

def GL(f,a,b,N):
    x,w = np.polynomial.legendre.leggauss(N)
    xp = 0.5*(b-a)*x + 0.5*(b+a)
    wp = 0.5*(b-a)*w
    return np.sum(wp*f(xp))

def func(x):
    return x**4 - 2.*x + 1

###problem1###
def simp_pre(f, a, h):
    return (1./3.)*h*(f(a)+4*f(a+h)+f(a+2.*h))

def simp_full(f, a, d, N):
    h = (d-a)/2./float(N)
    x = np.arange(a, d-0.1*h, 2.*h)
    return np.sum(simp_pre(f, x, h))
print(simp_full(func, 0., 2., 100))

def simp_2(f, a, b, eps):
    In = 0.
    h = (b-a)/100.0
    x1 = np.arange(a+h, b, 2*h)
    x2 = np.arange(a+2*h, b, 2*h)
    Sn = (1/3.)*(f(a) + f(b) + 2*np.sum(f(x2)))
    Tn = (2./3.)*np.sum(f(x1))
    I2n = h*(Sn + 2*Tn)
    while abs(I2n - In) > eps:
        In = I2n
        h = h/2.
        Sn = Sn + Tn
        x1 = np.arange(a+h, b, 2*h)
        Tn = (2./3.)*np.sum(f(x1))
        I2n = h*(Sn + 2*Tn)
    return I2n
print(simp_2(func, 0., 2., 1.e-6))

def simp_38(f, a, h):
    return (3./8.)*h*(f(a)+3*f(a+h)+3*f(a+2.*h)+f(a+3.*h))

def simp_full38(f, a, d, N):
    h = (d-a)/3./float(N)
    x = np.arange(a, d-0.1*h, 3.*h)
    return np.sum(simp_38(f, x, h))
print(simp_full38(func, 0., 2., 100))


def double_GL(f, a, b, d, u, N):
    x, w = np.polynomial.legendre.leggauss(N)
    xp = 0.5*(b-a)*x+0.5*(b+a)
    wp = 0.5*(b-a)*w
    y = []
    for i in xp:
        yp = 0.5*(u(i)-d(i))*x+0.5*(u(i)+d(i))
        ywp = 0.5*(u(i)-d(i))*w
        y.append(np.sum(f(i, yp)*ywp))
    return np.sum(wp*np.array(y))
def F1(x, y):
    return np.exp(y/x)
def u1(x):
    return x**2
def d1(x):
    return x**3
print(double_GL(F1, 0.1, 0.5, d1, u1, 90))

def F2(x, y):
    return y
def u2(x):
    return np.sqrt(1.-x**2)
def d2(x):
    return 0.
print(double_GL(F2, -1, 1, d2, u2, 90)*2./np.pi, 4./3./np.pi)


def F3(x, y):
    return x**2
def u3(y):
    return np.sqrt(1.-y**2)
def d3(y):
    return -np.sqrt(1.-y**2)
print(double_GL(F3, -1, 1, d3, u3, 90))


























