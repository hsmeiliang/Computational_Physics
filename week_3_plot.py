# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import math




#question_1
def mysum(N, g):
    k = 0.0
    for i in range(1, N+1, 1):
        k += g(i)
    return k
def myfunc(x):
        return -1*(-1)**x*1./x
def N_and_2N_method(myfunc, mysum, epsilon):
    N = 100
#    while 1:
#        if (mysum(2*N, myfunc)-mysum(N, myfunc)) < epsilon :
#            print(mysum(2*N, myfunc))
#            print(2*N)
#            break
#        else:
#            N*=2
    s0 = mysum(N, myfunc)
    s1 = mysum(2*N, myfunc)
    while abs(s0 - s1) > epsilon :
        N*=2
        s0 = s1
        s1 = mysum(2*N, myfunc)
        #print(2*N)
    print(mysum(2*N, myfunc))
    
epsilon = 1.e-4
N_and_2N_method(myfunc, mysum, epsilon)
####


#question_2
def myfactorial_v1(N):
    k = 1;
    for i in range(1,N+1):
      k *= i
    return k
def myfactorial_v2(N):
    if N == 0 : return 1
    else : return myfactorial_v2(N-1)*N
print(myfactorial_v1(10))
print(myfactorial_v2(10))
print(math.factorial(10))
####


#question_3
def myfunc_q3(x):
        return ((-1)**x)/(math.factorial(2*x))
N_and_2N_method(myfunc_q3, mysum, epsilon)
###



#question_4
a1 = [1, 4]
a2 = [2, 3]
a3 = np.array(a1)
a4 = np.array([2,3])
print(a1+a2)
print(a3+a4)
a1.append(3)
print(a1)
list1 = [3, -1, 2, -3, 1, 4, 5, 6]
print(len(list1))
###
print("question_5")


#question_5
i = 0
list_q5 = [3,-1,2,-3,1,8,5,6]
for i in list_q5:
    if i%2 == 0 : print(i)
###


#question_6
k = list_q5[0]
for i in list_q5:
    if i<k : k = i
print(k)
print(np.min(list_q5))
###

print("question_7")
#question_7
def Fib(x):
    if x == 1 : return 1
    elif x == 2 : return 1
    else : return Fib(x-1)+Fib(x-2)

f1, f2 = 1, 1
fib = [f1, f2]
while f1+f2<1000:
    f1, f2 = f2, f1+f2
    fib.append(f2)
print(fib)
###


#question_8
array8 = np.linspace(1, 10, 5)
# 起點 1 終點 10，分 5 等份
print(array8)
print("Sum of arr(uint8) : ", np.sum(array8, dtype = np.uint))  
print("Sum of arr(float32) : ", np.sum(array8, dtype = np.float))
print ("Is np.sum(arr).dtype == np.uint : ",np.sum(array8).dtype == np.uint) 
print ("Is np.sum(arr).dtype == np.float : ",np.sum(array8).dtype == np.float)  
print(np.arange(3))
print(np.arange(3.0))
print(np.arange(3,7,2))#like range
###


#question_9
from matplotlib import pyplot as plt
x = np.linspace(-10, 10, 1000)
y1 = x**2
y2 = x**3
plt.plot(x, y1, label = "$y = x^2$", color = "red", linewidth = 2)
plt.plot(x, y2, label = "$y = x^3$", color = "blue", linewidth = 1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("question_9")
plt.ylim(-100, 100)
plt.legend()
plt.show()

plt.subplot(2,2,1) # 第一行的左图
plt.plot(x, y1, label = "$y = x^2$", color = "red", linewidth = 2)
plt.subplot(2,2,2) # 第一行的右图
plt.plot(x, y2, label = "$y = x^3$", color = "blue", linewidth = 2)
plt.subplot(2,1,2) # 第二整行
plt.show()

plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.subplot(2,3,4)
plt.plot([0,1],[0,2])
plt.subplot(235)
plt.plot([0,1],[0,3])
plt.subplot(236)
plt.plot([0,1],[0,4])
plt.show()

plt.subplot(3,1,1)
plt.subplot(3,2,4)
plt.subplot(3,2,5)
plt.show()
###


#question_10
x = np.linspace(0, 10.*np.pi, 2001)
r = x**2
a, b = r*np.cos(x), r*np.sin(x)
plt.plot(a,b)
plt.show()

r = np.exp(np.cos(x))-2*np.cos(4*x)+(np.sin(x/12))**5
a, b = r*np.cos(x), r*np.sin(x)
plt.plot(a,b)
plt.savefig("./p.jpg")
plt.show()

###

t = np.linspace(0, 2*math.pi, 1000)
x1 = np.cos(t)
y1 = np.sin(t)
plt.plot(x1, y1)
plt.axis("equal")
plt.show() 

#question_11
