import numpy as np
import pylab as plt

def pre_RK4(f,a,h,y):
    k1 = h*f(y,a)
    k2 = h*f(y+k1/2.,a+h/2.)
    k3 = h*f(y+k2/2.,a+h/2.)
    k4 = h*f(y+k3,a+h)
    return a+h, y+(k1+2.*k2+2.*k3+k4)/6.

def AB4(f,a,b,h,y0):
    sol = []
    t = []
    sol.append(y0)
    t.append(a)
    a,y1 = pre_RK4(f,a,h,y0)
    sol.append(y1)
    t.append(a)
    a,y2 = pre_RK4(f,a,h,y1)
    sol.append(y2)
    t.append(a)
    a,y3 = pre_RK4(f,a,h,y2)
    sol.append(y3)
    t.append(a)     
  
    while a + h < b:
       y4 = y3 + h/24.0*(55.*f(y3,a) - 59.*f(y2,a-1.*h) + \
                     37.*f(y1,a-2.*h) - 9.*f(y0,a-3.*h))
       a = a + h
       sol.append(y4)
       t.append(a)
       y0 = y1
       y1 = y2
       y2 = y3
       y3 = y4
    y4 = y3 + (b-a)/24.0*(55.*f(y3,a) - 59.*f(y2,a-1.*h) + \
                    37.*f(y1,a-2.*h) - 9.*f(y0,a-3.*h)) 
    t.append(b)
    sol.append(y4)
    return np.array(t), np.array(sol)

def F(y,x): 
    return np.array([y[1],y[2],(-5.*x**2*y[2]-2.*x*y[1]+2.*y[0]+7.*x**(3./2.))/x**3])

U0 = np.array([10.6,-3.6,31.2])

t1,y1 = AB4(F,1.,10.,0.1,U0)

plt.plot(t1,y1[:,0])
plt.show()
