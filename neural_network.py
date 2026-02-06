# following "A Neural Network in 11 lines of Python (Part 1) article" by iamtrask

# Part 1: Tiny Toy Network
# neural network trained w/ backprop attempting to use input to predict output
# Backpropagation - measures statistics to make a model

import numpy as np
# numpy is linear algebra library

# sigmoid function - S-shaped/sigmoid curve
# sigmoid func maps number into probability value btwn 1 and 0
# output of sigmoid func can be used to create it's derivative (slope)
def nonlin(x,deriv=False):
# nonlinearity that maps sigmoid function
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
