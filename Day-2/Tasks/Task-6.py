#Sqaure Root of real Complex numbers.

import cmath

num = 1+2j

num_sqrt = cmath.sqrt(num)

print('The SquareRoot of {0} is {1:0.3f}+{2:0.3f}j'.format(num,num_sqrt.real,num_sqrt.imag))