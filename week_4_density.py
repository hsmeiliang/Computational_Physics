# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import scipy as sci
from scipy import special

###problem_1
def Legendrepoly(x, n):
    if n == 0: return np.ones(len(x))
    elif n == 1: return x
    else : return ((2*n-1)*x*Legendrepoly(x,n-1)-(n-1)*Legendrepoly(x,n-2))/n
x = np.linspace(-1., 1., 1000)
y1 = Legendrepoly(x, 2)
y2 = Legendrepoly(x, 3)
y3 = Legendrepoly(x, 4)
y4 = Legendrepoly(x, 5)
plt.plot(x, y1, color = "red", linewidth = 2)
plt.plot(x, y2, color = "blue", linewidth = 2)
plt.plot(x, y3, color = "black", linewidth = 2)
plt.plot(x, y4, color = "green", linewidth = 2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("problem_1")

plt.legend()
plt.show()
###

###problem_2
def Hermitepoly(x, n):
    if n == 0: return np.ones(len(x))
    elif n == 1: return 2*x
    else : return 2*x*Hermitepoly(x, n-1)-2*(n-1)*Hermitepoly(x, n-2)
x = np.linspace(-5., 5., 1000)
y1 = Hermitepoly(x, 2)
y2 = Hermitepoly(x, 3)
y3 = Hermitepoly(x, 4)
y4 = Hermitepoly(x, 5)
plt.plot(x, y1, label = "$y = H_2$", color = "red", linewidth = 2)
plt.plot(x, y2, label = "$y = H_3$", color = "blue", linewidth = 2)
plt.plot(x, y3, label = "$y = H_4$", color = "black", linewidth = 2)
plt.plot(x, y4, label = "$y = H_5$", color = "green", linewidth = 2)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-200, 200)
plt.title("problem_2")

plt.legend()
plt.show()
###

###problem_3
def associ_Legendrepoly(x, n, k):
    if n == 0: return np.ones(len(x))
    elif n == 1: return -x+k+1
    else : return ((2*n+k-1-x)*associ_Legendrepoly(x,n-1, k)-(n+k-1)*associ_Legendrepoly(x,n-2, k))/n
r = np.linspace(0., 50., 5000)
n = 3
l0 = 0
l1 = 1
l2 = 2
def radial(n, l, r):
    amp = np.sqrt(((2./n)**3)*math.factorial(n-1-l)/2./n/(math.factorial(n+l))**3)
    return amp*(2.*r/n)**l*np.exp(-r/n)*associ_Legendrepoly(2*r/n, n-1-l, 2*l+1)
R0 = r**2*(radial(n, l0, r))**2
R1 = r**2*(radial(n, l1, r))**2
R2 = r**2*(radial(n, l2, r))**2
plt.plot(r, R0, label = "$y = R0$", color = "blue", linewidth = 2)
plt.plot(r, 50*R1, label = "$y = R1$", color = "red", linewidth = 2)
plt.plot(r, 1000*R2, label = "$y = R2$", color = "green", linewidth = 2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("problem_2")
plt.legend()
plt.show()
###

###problem_4

a = np.loadtxt('potential.txt')
print(np.shape(a))
plt.imshow(a)
plt.show()
plt.imshow(a, origin = "lower")
plt.show()
###

###problem_5
def PHD(x, y, z, n, l, m):
    r = np.sqrt(x**2+y**2+z**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, np.sqrt(x**2+y**2))
    amp = np.sqrt(((2./n)**3)*math.factorial(n-1-l)/2./n/(math.factorial(n+l))**3)
    return amp*(2.*r/n)**l*np.exp(-r/n)*sci.special.assoc_laguerre(2*r/n, n-l-1, 2*l+1)*sci.special.sph_harm(m, l, theta, phi)*r*np.sqrt(np.sin(theta))
N = 200
x = np.linspace(-20., 20., N)
y = np.linspace(-20., 20., N)
density = []
for i in x:
    for j in y:
        density.append(abs(PHD(i, j, 1, 3, 2, 0))**2)
density = np.reshape(density, (N, N))
plt.imshow(density)
plt.show()










