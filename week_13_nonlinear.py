import numpy as np
import pylab as plt

def pre_RK4(F,a,h,U0):
    K1 = F(a,U0)
    K2 = F(a+0.5*h,U0+0.5*K1)
    K3 = F(a+0.5*h,U0+0.5*K2)
    K4 = F(a+h,U0+K3)
    return a+h, U0 + h*(K1+2.0*K2+2.0*K3+K4)/6.0

def RK4(F,a,b,h,U0):
    sol = []
    t = []
    t.append(a)
    sol.append(U0)

    while a + h < b:
       a,U0 = pre_RK4(F,a,h,U0)
       sol.append(U0)
       t.append(a)
    a,U0 = pre_RK4(F,a,b-a,U0)
    sol.append(U0)
    t.append(a)
    return np.array(t), np.array(sol)

def fixed_pt(f,x,epsilon):
    x1 = f(x)
    while abs(x1-x) > epsilon:
      x = x1
      x1 = f(x)
    return 0.5*(x+x1)

t = np.arange(0.01, 2, 0.05)
m = []
for i in t:
    def ms1(x):
        return np.tanh(x/i)
    m.append(fixed_pt(ms1, i, 1.e-8))
plt.plot(t, m)
plt.show()

def bis(f, a, b, epsilon):
    while abs(f(a)-f(b)) > epsilon:
        c = (a+b)/2.
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return c

def func(x):
    return np.cos(x)-x
a = bis(func, 0., 1., 1.e-8)
print(func(a))

def func3(x):
    return (1.-5.*x)*np.exp(1./x)+5.*x
x = np.arange(0.15, 0.3, 0.1)
plt.plot(x, func3(x))
plt.show()

kb = 1.3806488e-23
c = 2.99792458e8
h = 6.62606957e-34
a = bis(func3, 0.19, 0.21, 1.e-8)
print(a, a*h*c/kb/(500.e-9))

def newton(f, df, x, esp):
    x1 = x - f(x)/df(x)
    while abs(x1-x) > esp:
        x = x1
        x1 = x - f(x)/df(x)
    return x1
def df1(x):
    return -np.sin(x)-1
print(newton(func, df1, 1., 1.e-8))

def secant(f, x, eps):
    h = 0.1
    x1 = x-h*f(x)/(f(x+h)-f(x))
    while abs(x-x1)>eps:
        x= x1
        x1 = x-h*f(x)/(f(x+h)-f(x))
    return x1
print(secant(func, 1., 1.e-8))


def AB4(f, a, b, h, y0):
    sol = []
    t = []
    sol.append(y0)
    t.append(a)
    a, y1 = pre_RK4(f, a, h, y0)
    sol.append(y1)
    t.append(a)
    a, y2 = pre_RK4(f, a, h, y1)
    sol.append(y2)
    t.append(a)
    a, y3 = pre_RK4(f, a, h, y2)
    sol.append(y3)
    t.append(a)
    while a+h < b:
        y = y3+h/24.*(55.*f(y3, a)-59.*f(y2, a-h)+37.*f(y1, a-2*h)-9.*f(y0, a-3*h))
        a = a+h
        sol.append(y)
        t.append(a)
        y0 = y1
        y1 = y2
        y2 = y3
        y3 = y
    h = b-a
    y = y3 +y3+h/24.*(55.*f(y3, a)-59.*f(y2, a-h)+37.*f(y1, a-2*h)-9.*f(y0, a-3*h))
    sol.append(y)
    t.append(h)
    return np.array(t), np.array(sol)

def f5(y, x):
    return np.array([y[1], y[2], -5*x**2*y[2]/x**3-2*x*y[1]/x**3+2*y[0]/x**3+7*x**(1.5)/x**3])
u0 = np.array([10.6, -3.6, 31.2])
t, y = AB4(f5, 1., 10., 0.001, u0)
plt.plot(t, y[:, 0])
plt.show()








