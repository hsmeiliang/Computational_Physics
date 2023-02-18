import numpy as np
import matplotlib.pyplot as plt

def EQ(c, threshold):
    z = complex(0, 0)
    for iteration in range(threshold):
        z = (z*z) + c
        if abs(z) > 2:
            break
    return iteration

def mandelbrot(x1, x2, y1, y2, threshold, N):
    
    realAxis = np.linspace(x1, x2, N)
    imagAxis = np.linspace(y1, y2, N)
    
    realLen = len(realAxis)
    imagLen = len(imagAxis)
    arr = np.empty((realLen, imagLen))
    for ix in range(realLen):
        for iy in range(imagLen):
            cx = realAxis[ix]
            cy = imagAxis[iy]
            c = complex(cx, cy)
            arr[ix, iy] = EQ(c, threshold)
    plt.imshow(arr.T)
    plt.show()
    
#mandelbrot(-1.5, 1., -1.5, 1., 100, 500)
mandelbrot(-1.5, -0.5, -0.5, 0.5, 100, 500)
mandelbrot(-1.5, -1., -0.25, 0.25, 100, 500)
mandelbrot(-1.5, -1.25, -0.125, 0.125, 100, 500)
mandelbrot(-1.4375, -1.375, -0.03625, 0.03625, 100, 500)
#mandelbrot(-1.5, -1.375, -0.0625, 0.0625, 100, 500)

#mandelbrot(-0.22, -0.21, -0.7, -0.69, 100, 500)
#mandelbrot(-0.22, -0.219, -0.7, -0.699, 100, 500)
#mandelbrot(-0.22, -0.2195, -0.7, -0.6995, 100, 500)
#mandelbrot(-0.22, -0.21975, -0.69975, -0.6995, 100, 500)
