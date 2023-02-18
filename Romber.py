import numpy as np
import scipy.special
import math
import pylab as plt

def func(x):
    return x**4 - 2.*x + 1

def romber_int(f, a, b, max_step, eps):
    h = (b-a)
    R1 = np.zeros(len(np.linspace(0, max_step, max_step+1)))
    R2 = np.zeros(len(np.linspace(0, max_step, max_step+1)))
    R1[0] = (f(a) + f(b)) * h * 0.5
    
    for i in range(1, max_step, 1):
        h = h/2.
        c = 0
        ep = 2**(i-1)
        for j in range(1, ep+1, 1):
            c = c + f(a + (2*j-1)*h)
        R2[0] = h*c +0.5*R1[0]
        for j in range(1, i+1, 1):
            nk = 4**j
            R2[j] = (nk * R2[j-1] - R1[j-1])/(nk - 1)
        if i > 1:
            if abs(R1[i-1] - R2[i]) < eps:
                return R2[i-1]
        Rt = R1
        R1 = R2
        R2 = Rt
    return R1[max_step - 1]

print(romber_int(func, 0., 2., 100, 1.e-6))