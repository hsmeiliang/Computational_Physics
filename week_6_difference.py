import numpy as np
import pylab as plt
import cmath
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy as sci
from scipy import special
import time


###problem1
mylist = [16, 24, 32, 40]
for i in mylist:
    a = np.loadtxt('data_set_2_L{0}.txt'.format(i))
    my_mean = np.average(a[:,4])
    my_uncertainty = np.std(a[:,4])/np.sqrt(len(a)-1)
    myfile = open('result.dat', 'a')
    myfile.write(" %.8f " % a[1, 1])
    myfile.write(" %.8f " % my_mean)
    myfile.write(" %.8f \n" % my_uncertainty)
    myfile.close()
a3 = np.loadtxt('result.dat')
x = a3[:, 0]
y = a3[:, 1]
dy = a3[:, 2]
plt.errorbar(x, y, dy, marker = 'o', markersize = 2, color = 'blue', linestyle = '--', capsize = 6)
plt.xlabel(r'L', fontsize = 20)
plt.ylabel(r'$\langle (m_s^z)^2\rangle$', fontsize = 20)
plt.show()
###


###problem2
def func(x):
    return x**(0.5)
def forwarddiff(f, x, h):
    return (f(x-h)-2*f(x)+f(x+h))/(h**2)
def Cendiff(f, x, h):
    return (2*f(x)-5*f(x+h)+4*f(x+2*h)-f(x+3*h))/(h**2)
def fivediff(f, x, h):
    return (-f(x-2*h)+16.*f(x-h)-40.*f(x)+16*f(x+h)-f(x+2*h))/(12.*h**2)

x = 2
y = np.sqrt(x)
err = []
h = np.array([0.2, 0.1, 0.05, 0.01, 0.001, 1.e-4, 1.e-8, 1.e-10, 1.e-12, 1.e-14, 1.e-16])
result1 = np.abs((forwarddiff(func, x, h)-y)/y)
plt.plot(h, result1, 'sc', label = 'forward', color = 'blue')
result2 = np.abs((Cendiff(func, x, h)-y)/y)
plt.plot(h, result2, 'ob', label = 'Central', color = 'black')
result3 = np.abs((fivediff(func, x, h)-y)/y)
plt.plot(h, result3, 'vr', label = 'fivepoint', color = 'red')
#plt.ylim(1.e-5, 1e40)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
















