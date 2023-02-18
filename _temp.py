import numpy as np
import matplotlib.pyplot as plt

julia_C = -0 + 0j

def julia_set(x, y):
   z = np.array(x + 1j * y)
   r = np.zeros(z.shape)
   m = np.ones(z.shape, dtype=bool)
   for i in range(24):
       z[m] = z[m] ** 2 + julia_C
       m = np.abs(z) < 2
       r += m
   return r

def mandelbrot_set(x, y):
   c = np.array(x + 1j * y)
   z = np.zeros(c.shape, dtype=complex)
   r = np.ones(c.shape)
   m = np.ones(c.shape, dtype=bool)
   for i in range(50):
       z[m] = z[m] ** 2 + c[m]
       m = np.abs(z) < 2
       r += m
   return r

def complex_str(c):
   return np.array_str(np.array([julia_C]), suppress_small=True, precision=3)

def grid(width, offset, n):
   x = np.linspace(-width + offset, width + offset,n)
   y = np.linspace(-width, width, n)
   return np.meshgrid(x,y), (x.min(), x.max(), y.min(), y.max())
i = 1
j = -1


for a in range(0, 5, 1):
    (X, Y), extent = grid(i, j, 1000)
    plt.imshow(mandelbrot_set(X,Y), extent=extent)
    plt.show()
    i = i*0.5
    j = -1-0.25*a