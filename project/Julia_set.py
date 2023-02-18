import numpy as np
import matplotlib.pyplot as plt

def EQ(Z, c, threshold):
    for iteration in range(threshold):
        Z = (Z*Z) + c
        if abs(Z) > 2:
            break
    return iteration

def julia(x1, x2, y1, y2, c, threshold, N):
    
    realAxis = np.linspace(x1, x2, N)
    imagAxis = np.linspace(y1, y2, N)
    
    realLen = len(realAxis)
    imagLen = len(imagAxis)
    arr = np.empty((realLen, imagLen))
    for ix in range(realLen):
        for iy in range(imagLen):
            Zx = realAxis[ix]
            Zy = imagAxis[iy]
            Z = complex(Zx, Zy)
            arr[ix, iy] = EQ(Z, c, threshold)
    plt.imshow(arr.T)
    plt.show()



c = -0.4 + 0.6j
julia(-2., 2., -2., 2., c, 100, 500)

c = 0.285 + 0.01j
julia(-2., 2., -2., 2., c, 100, 500)

c = -0.70176 - 0.3842j
julia(-2., 2., -2., 2., c, 100, 500)

c = -0.835 - 0.2321j
julia(-2., 2., -2., 2., c, 100, 500)

