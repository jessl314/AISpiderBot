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

# input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# output dataset
y = np.array([[0,0,1,1]]).T

# input/output dataset matrix where each row is a training example

# seed random num to make calculation
# deterministic (good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
    # forward propagation
    l0 = X
    # l0 is first layer of network, specified by input data
    l1 = nonlin(np.dot(l0,syn0))
    # l1 is second layer, aka hidden layer

    # how much missed
    l1_error = y - l1

    # multiply how much missed by
    # slope (derivative) of sigmoid at values in l1
    l1_delta = l1_error * nonlin(l1,True)
    # This line does the most

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print(f"Output after training:\n{l1}")