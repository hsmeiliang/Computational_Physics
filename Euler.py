import numpy as np
import pylab as plt
def pre_imp_euler(f, x0, y0, h):
    yt = y0 + h*f(x0, y0)
    return x0 + h, y0 + h/2.*(f(x0, y0)+f(x0+h, yt))

def imp_euler(f, x0, xf, N, y0):
    h = (xf - x0)/float(N)
    sol = []
    x = []
    sol.append(y0)
    x.append(x0)
    while x0+h < xf:
        x0, y0 = pre_imp_euler(f, x0, y0, h)
        sol.append(y0)
        x.append(x0)
    x0, y0 = pre_imp_euler(f, x0, sol[-1], xf-x0)
    sol.append(y0)
    x.append(x0)
    return np.array(x), np.array(sol)

def func(t, y):
    return np.array([y[1], -1/2*y[1]-1/2*np.sin(y[0])])
z = [1.0, 0.0]
x, y = imp_euler(func, 0.0, 360.0, 100, z)
plt.plot(x, y[:, 0])
plt.plot(x, y[:, 1])
plt.show()











