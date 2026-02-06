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

# initializes input dataset as numpy matrix
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
# each column -> one input nodes
# 3 input nodes to network and 4 training examples

# initializes output dataset
y = np.array([[0,0,1,1]]).T
# single row, 4 columns
# T is transpose function
# after transpose, y matrix has 4 rows, 1 column
# each row is training example
# each column is output node
# network has 3 inputs and 1 output


# seed random num to make calculation
# deterministic (good practice):
# randomly distributes in same way each time trained
# easier to see how changes affect network
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
# "synapse zero" 
# since 2 layers, only need one matrix of weights to connect
# dimension is (3,1) because 3 inputs, 1 output
# l0 of size 3, l1 of size 1
# want to connect every node in l0 to every node in l1
# best practice to have mean of zero in weight initialization

# iterates over training code to optimize network to dataset
for iter in range(10000):
    # forward propagation
    l0 = X
    # l0 is first layer of network, specified by input data
    # process all at same time "full batch training"
    l1 = nonlin(np.dot(l0,syn0))
    # l1 is second layer, aka hidden layer

    # how much missed
    l1_error = y - l1

    # multiply how much missed by
    # slope (derivative) of sigmoid at values in l1
    l1_delta = l1_error * nonlin(l1,True)
    # secret sauce
    ''' 
    nonline(l1,True)
    code generates slope of l1
    multiplying them 'elementwise'
    returns (4,1) matrix l1_delta w/ the multiplied values
    '''
    '''
    l1_error is (4,1) matrix
    nonlin(l1,True) returns (4,1) matrix w/ multiplied values
    multiplying slopes by error -> reduces error of high condifence predictions
    if slope was shallow, neetwork had very high or low value, network confident
    leaves confident predictions alone by multplying # close to 
    updates 'wishy-washy' (x=0,y=0.5) predictions most heavily 
    '''
    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    # small error and small slope means very small update
    # computes weight updates for each weight for each training example
    # sums them, and update the line
    

print(f"Output after training:\n{l1}")
# When input & output are 1, increase weight
# when input is 1 & output is 0, decrease weight