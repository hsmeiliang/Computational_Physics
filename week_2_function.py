# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:09:33 2019

@author: user
"""



a = input('名字:')
# a is a string(py3.0) but is a num in py2.0
print('Hello, ', a)

en = float(input('english:'))
math = float(input('math:'))
ch = float(input('ch:'))
print('ave, ', (en+math+ch)/3)

num = int(input('input a positive integer:'))
if num%2 == 0 : print('The integer is even.')
else : print('The integer is odd.')

def f(x) : return x**3
print(f(3))

k = 0.
for i in range(1, 10001, 1):
    k += (1./i**2)
print(k)

def mysum(N, g):
    k = 0.0
    for i in range(1, N+1, 1):
        k += g(i)
    return k
def myfunc(x):
    return 1./(2*x-1)**3
print(mysum(10000, myfunc))

i = 0
while i<10 :
    if(i%2 == 0) : print(i)
    i += 1

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
        print(2*N)
    print(mysum(2*N, myfunc))
    
epsilon = 1.e-4
N_and_2N_method(myfunc, mysum, epsilon)

def myfactorial(N):
    k = 1;
    for i in range(1,N+1):
      k *= i
    return k
def myfactorial_v2(N):
    if N == 0 : return 1
    else : return myfactorial_v2(N-1)*N
def Catalan_number(N):
    if N == 0 : return 1.0
    else : return Catalan_number(N-1)*(4*N-2)/(N+1)
print(Catalan_number(10))

def gcd(m, n) :
    if n == 0 : return m
    else : return gcd(n, m%n)
print(gcd(1144, 9240))


import numpy as np
a1 = [1, 4]
a2 = [2, 3]
a3 = np.array(a1)
a4 = np.array([2,3])

print(a1+a2)
print(a3+a4)

a1.append(3)
print(a1)

list1 = [3, -1, 2, -3, 1, 4, 5, 6]
for i in list1:
    if i%2 == 0 : print(i)
k = list1[0]
for i in list1:
    if list1[i] < k : k = list1[i]
print(k)
print(len(list1))







