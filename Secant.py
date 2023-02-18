import numpy as np
import pylab as plt

z0 = 8.
def func1(z):
    return np.arctan(np.sqrt((z0/z)**2-1.))-z

def secant(f, x, eps):
    h = 0.1
    x1 = x-h*f(x)/(f(x+h)-f(x))
    while abs(x-x1)>eps:
        x= x1
        x1 = x-h*f(x)/(f(x+h)-f(x))
    return x1

def func2(z):
    return np.arctan(np.sqrt((z0/z)**2-1.))-z/(z-secant(func1, 1, 1.e-8))

def func3(z):
    return np.arctan(np.sqrt((z0/z)**2-1.))-z/(z-secant(func2, 1, 1.e-8))

print(secant(func1, 1., 1.e-8))
print(secant(func2, 1., 1.e-8))
print(secant(func3, 1., 1.e-8))