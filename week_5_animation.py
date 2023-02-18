# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:04:42 2019

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


###problem_1
def associ_Legendrepoly(x, n, k):
    if n == 0: return 1.
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

###problem_2
def PHD(x, y, z, n, l, m):
    r = np.sqrt(x**2+y**2+z**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    amp = np.sqrt(((2./n)**3)*math.factorial(n-1-l)/2./n/(math.factorial(n+l))**3)
    return amp*(2.*r/n)**l*np.exp(-r/n)*sci.special.assoc_laguerre(2*r/n, n-l-1, 2*l+1)*sci.special.sph_harm(m, l, theta, phi)*r*np.sqrt(np.sin(theta))
#    return amp*(2.*r/n)**l*np.exp(-r/n)*associ_Legendrepoly(2*r/n, n-l-1, 2*l+1)*sci.special.sph_harm(m, l, theta, phi)*r*np.sqrt(np.sin(theta))
N = 200
x = np.linspace(-20., 20., N)
y = np.linspace(-20., 20., N)
density = []
for i in x:
    for j in y:
        density.append(abs(PHD(i, j, 1.e-12,4,3,1))**2)
density = np.reshape(density, (N, N))
plt.imshow(density)
plt.show()
###

###problem_4
a3 = np.loadtxt('stm.txt')
print(np.shape(a3))
depth = np.max(abs(a3))
num = np.shape(a3)[0]
num1 = np.shape(a3)[1]
x3 = np.linspace(-4., 4., num1)
y3 = np.linspace(-2., 2., num)
Y3, X3 = np.meshgrid(x3, y3)
plt.imshow(a3)
plt.hot()
#Greys,Blues,Greens,Purples,gray, binary, spring, summer, winter,...
plt.show()
x = np.linspace(1, 6, 6)
y = np.linspace(1, 6, 6)
Y, X = np.meshgrid(x, y)
print(X,Y)
###

fig3 = plt.figure()
ax3 = Axes3D(fig3)
surf3 = ax3.plot_surface(X3,Y3,a3,rstride=1, cstride=1,cmap=cm.coolwarm,linewidth=0,alpha=1.0)
ax3.contourf(X3,Y3,a3, zdir='z',offset = -2.5*depth, cmap = cm.coolwarm)
fig3.colorbar(surf3,shrink=0.75)
ax3.set_zlim(-2.5*depth,depth)
ax3.view_init(15,20)
plt.show()


###problem_5
a5 = np.loadtxt('potential1.txt')
print(np.shape(a5))
depth = np.max(abs(a5))
num = np.shape(a5)[0]
num1 = np.shape(a5)[1]
x5 = np.linspace(-4., 4., num1)
y5 = np.linspace(-2., 2., num)
Y5, X5 = np.meshgrid(x5, y5)
plt.imshow(a5)
plt.hot()
#Greys,Blues,Greens,Purples,gray, binary, spring, summer, winter,...
plt.show()
#x = np.linspace(1, 6, 6)
#y = np.linspace(1, 6, 6)
#Y, X = np.meshgrid(x, y)
#print(X,Y)
fig5 = plt.figure()
ax5 = Axes3D(fig5)
surf5 = ax5.plot_surface(X5,Y5,a5,rstride=1, cstride=1,cmap=cm.coolwarm,linewidth=0,alpha=1.0)
ax5.contourf(X5,Y5,a5, zdir='z',offset = -2.5*depth, cmap = cm.coolwarm)
fig5.colorbar(surf5,shrink=0.75)
ax5.set_zlim(-2.5*depth,depth)
ax5.view_init(15,20)
plt.show()
###
N1 = np.shape(a5)[0]
x5 = np.linspace(0, N1-1, N1)
y5 = np.linspace(0, N1-1, N1)
Y5, X5 = np.mgrid[0:N1, 0:N1]
fy, fx = np.gradient(a5)
plt.streamplot(X5, Y5, -fx, fy)
plt.show()




###problem_6
Lx = np.loadtxt('LorenzX.txt')
Ly = np.loadtxt('LorenzY.txt')
Lz = np.loadtxt('LorenzZ.txt')
fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
ax3.plot(Lx, Ly, Lz, lw=0.5)
ax3.set_xlabel("X Axis")
ax3.set_ylabel("Y Axis")
ax3.set_zlabel("Z Axis")
ax3.set_title("Lorenz Attractor")
plt.show()
###

###problem_9
a6 = np.loadtxt('wave.txt')
x = np.linspace(0.,1.0,np.shape(a6)[1])
plt.plot(x,a6[0,:])
plt.show()
plt.plot(x,a6[2,:])
plt.show()
plt.plot(x,a6[5,:])
plt.show()
plt.plot(x,a6[10,:])
plt.show()
plt.plot(x,a6[20,:])
plt.show()

def animation_1D_wave(p,L1,L2,zm=0,zm1=0):
    num_x = np.shape(p)[1]
    num_t = np.shape(p)[0]
    x1 = np.linspace(L1,L2,num_x)
    up = np.max(np.abs(p[0,:]))
    down = np.max(np.abs(p[0,:]))
    plt.ylim(down-zm*abs(down), up + zm1*abs(up))
    line, = plt.plot(x1,p[0,:])
    for i in range(num_t):
      line.set_ydata(p[i,:])
      plt.draw()
      plt.pause(0.01)

animation_1D_wave(a6,0.,1.0,3,1)





























