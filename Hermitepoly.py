#2019_10_14_HW
#40541204S_梁湘梅

import numpy as np
import pylab as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def Hermitepoly(x, n):
    if n == 0: return 1.
    elif n == 1: return 2*x
    else : return 2*x*Hermitepoly(x, n-1)-2*(n-1)*Hermitepoly(x, n-2)
def QHO_1D(x, n):
    return ((1./(np.pi*(2**(2*n))*((math.factorial(n))**2)))**(1./4))*np.exp(-(x**2)/2)*Hermitepoly(x, n)
def QHO_2D(x, n, y, m):
    return QHO_1D(x, n)*QHO_1D(y, m)

n, m = 5, 3
N = 200
x = np.linspace(-5., 5., N)
y = np.linspace(-5., 5., N)

density = []
for i in x:
    for j in y:
        density.append(abs(QHO_2D(i, n, j, m))**2)

density = np.reshape(density, (N, N))
plt.imshow(density)
plt.show()

a5 = density

depth = np.max(abs(a5))
num = np.shape(a5)[0]
num1 = np.shape(a5)[1]
Y5, X5 = np.meshgrid(x, y)
fig5 = plt.figure()
ax5 = Axes3D(fig5)
surf5 = ax5.plot_surface(X5,Y5,a5,rstride=1, cstride=1,cmap=cm.coolwarm,linewidth=0,alpha=1.0)
ax5.contourf(X5,Y5,a5, zdir='z',offset = -2*depth, cmap = cm.coolwarm)
fig5.colorbar(surf5,shrink=0.75)
ax5.set_zlim(-2*depth,depth)
plt.show()



