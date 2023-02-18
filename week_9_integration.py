import numpy as np
import scipy.special
import math
import pylab as plt
def GL(f,a,b,N):
    x,w = np.polynomial.legendre.leggauss(N)
    xp = 0.5*(b-a)*x + 0.5*(b+a)
    wp = 0.5*(b-a)*w
    return np.sum(wp*f(xp))

def trapz_2(f, a, b, eps):
    s0 = 0.
    h = (b-a)/100.0
    x = np.arange(a+h, b, h)
    s1 = h*(np.sum(f(x)) + 0.5*(f(a) + f(b)))
    while abs(s1 - s0) > eps:
        s0 = s1
        x = np.arange(a+h/2., b, h)
        s1 = s0/2. + np.sum(f(x))*h/2.
        h = h/2.
    return s1
def func(x):
    return x**4 - 2.*x + 1
print(trapz_2(func, 0., 2., 1.e-6))

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

def improper_int(f):
    def g(z):
        return f(z/(1-z))/(1-z)**2
    return GL(g, 0., 1., 50)

def f2(x):
    return np.exp(-x**2)
def f3(x):
    return x**3/(np.exp(x)-1)
print(improper_int(f2), (np.pi)**0.5/2.)
c = 299792458.0
h = 1.05457173e-34
k = 1.38e-23
print(improper_int(f3)*k**4/(4*(np.pi**2)*(c**2)*(h**3)))

def improper_int2(f):
    def g(z):
        return f(z/(1-z**2))*(1+z**2)/(1-z**2)**2
    return GL(g, -1., 1., 90)
print(improper_int2(f2), (np.pi)**0.5)

def Phi(n, x):
    return 1./np.sqrt(2**n*math.factorial(n)*math.pi**0.5)*np.exp(-x**2/2.)*scipy.special.eval_hermite(n, x)
x = np.linspace(-10, 10, 1000)
y = Phi(30, x)
plt.plot(x, y)
plt.show()

def expect(x):
    return Phi(5, x)**2*x**2
print(improper_int2(expect))


def ex11(n):
    def g(x):
        return Phi(n, x)**2*x**4
    return improper_int2(g)
print(ex11(4), 3/4.*(1+2*4+2*4**2))
## n<15 because of GL
















