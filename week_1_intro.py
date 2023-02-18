# need to import required and useful libraries
from numpy import *
import numpy as np
import math
from math import *

a = 5
b = 3
c = 5.
d = 3.

# integer/integer is an integer in python 2.7
print(a*b, a+b, a-b, a/b, a%b, a**2, float(a)/b, c/d, int(c/d)) 

print ('My phone number is', 6028)
string = 'A'
print ('My phone number is {} and room is'.format(6028) + '', str(string) + '{}'.format(203))
print ('The Pi is %.10f'%math.pi)
print ('My room is {}{}'.format('A',203))

x = input('Please enter an integer number: ')
print ('The number you enter is',x)
y = raw_input('Please enter your name: ')
print ('Hello,',y)

# print out integer from 1 to 9
# range(a,N,b) starts with a and then a + b, a + 2b, a + 3b ... until 
# a + mb <= N - 1 
 
for i in range(10):
  print (i)

# print out even number < 10 and starts with 2
for i in range(2,10,2):
  print (i)

# same as above
for i in range(2,10):
   if i%2 == 0 : print (i)

# same as above
i = 2
while i < 10:
  if i%2 == 0: print (i) 
  i += 1

# same as above
i = 2
while i < 10:
  if i%2 == 0: print (i)
  if i > 8: break
  i += 1

for i in range(101):
   if i%3 == 0: print (i)
   elif i%5 == 0: print (i)
   else: print ('{} does not have 3 or 5 as its factors '.format(i))

i = 4
prime_number = []
while i < 100:
  i += 1
  k = 0
  for j in range(2,i):
    if i%j == 0: k = 1
  if k == 0: prime_number.append(i)
print (prime_number)

def f(x): return x**2

print (f(3))

def myfactorial(i):
    k = 1;
    for j in range(1,i+1):
      k = k*j
    return k

print myfactorial(5)

def lcm(a,b):
    if a > b:
      c = a
    else: c = b
    k = a*b; 
    for j in range(c,a*b):
      if j%a == 0 and j%b == 0:
         k = j; break 
    return k

print (lcm(9,15))

def gcf(a,b):
    if a > b:
      c = b
    else: c = a
    d = 1
    j = 1
    while j < c:
      j += 1
      if a%j == 0 and b%j == 0:
        d *= j
        c = c/j
        a = a/j; b = b/j         
        j = 1 
    return d
print (gcf(24,50))

def myfactorial1(n):
    if n == 0: return 1.
    elif n == 1: return 1.
    else: return n*myfactorial1(n-1)

print (myfactorial1(5))
   

def xy_polar(x,y):
    theta = atan(y/x)
    r = sqrt(x**2 + y**2)
    return r, theta

print (xy_polar(1.,1.))
print (sqrt(2), math.pi/4.) 

def mysum_1(N):
    s = 1.
    for i in range(2,N+1):     
      s += 1./i**2
    return s

print (mysum_1(50))


def Catalan(n):
    if n == 0: return 1.0
    else: return (4.0*(n-1)+2.0)/((n-1)+2.0)*Catalan(n-1)

k = 0
while(Catalan(k) < 10**8):
  print (int(Catalan(k)))
  k += 1

k1 = input("Please enter the first positive integer: ")
k2 = input("Te second positive integer: ")
g2 = gcf(k1,k2)

print ("The great common factor of {} and {} is {}".format(k1,k2,g2))

x = raw_input("Please enter your name: ")
print ('Hello,', x)

A = [1,1,3]
A1 = [4,5]
A2 = [2,4,5]
B = np.array([1,1,3])
B1 = np.array([4,5,3])
print (2*A)

print (2*B)

print A+A1
print A+A2
print B+B1
