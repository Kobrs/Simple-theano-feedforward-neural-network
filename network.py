import theano
from theano import tensor as T
from theano.ifelse import ifelse
import numpy as np 
from random import random
import matplotlib.pyplot as plt
import pylab
from progressbar import *

# This is useful if you tweak something and theano's error isn't helpful
# theano.config.optimizer='fast_compile'

input_layer_size = 400
hidden_layer_size = 100
output_layer_size = 4

learning_rate = 0.1
lambda1 = 0.00
lambda2 = 0.001
iterations = 100

rng = np.random.RandomState(1234)

# sigmoid function
def g(t):
    return 1/(1+T.exp(-t))

# create dataset
def get_dataset():
    # uncomment whichever dataset you like to use
    
    # DATASET (XOR gate):
    # X = np.array([
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1]
    # ])

    # Label vector (or rather matrix of shape = (N,1)
    # Y = np.array([[0, 1, 1, 0]]).T

    # DATASET (random generated):
    N = 100

    X = rng.randn(N, input_layer_size)
    Y = rng.randint(size=(N, output_layer_size), low=0, high=2)

    return X,Y

def weights_init(input_layer_size, hidden_layer_size, 
                output_layer_size, activations):

    w1 = np.asarray(
        rng.uniform(            
            low=-np.sqrt(6. / (input_layer_size + hidden_layer_size)),
            high=np.sqrt(6. / (input_layer_size + hidden_layer_size)),
            size=(input_layer_size, hidden_layer_size)),
        dtype = theano.config.floatX)

    w2 = np.asarray(
        rng.uniform(            
            low=-np.sqrt(6. / (hidden_layer_size + output_layer_size)),
            high=np.sqrt(6. / (hidden_layer_size + output_layer_size)),
            size=(hidden_layer_size,  output_layer_size)),
        dtype = theano.config.floatX)
  
        # Since we are using sigmoid function, we should multiply our weights by 4
    w1 = w1 * 4
    w2 = w2 * 4

    w1 = theano.shared(w1, name='w1')
    w2 = theano.shared(w2, name='w2')    
    b1 = theano.shared(np.zeros((hidden_layer_size,)), name='b1')
    b2 = theano.shared(np.zeros((output_layer_size,)), name='b2')
    
    return w1, w2, b1, b2

# Feedforward function:
def predict_func(X, w1, w2, b1, b2):

    z2 = T.dot(X, w1)+b1
    a2 = g(z2)

    z3 = T.dot(a2, w2)+b2
    h = g(z3)

    return h


def cost_func(X, Y, h, w1, w2):
    L1 = T.sum(abs(w1)) + T.sum(abs(w2))
    L2 = T.sum(w1**2) + T.sum(w2**2)

    cost = (-Y*T.log(h) - (1-Y)*T.log(1-h)).sum()
    cost = cost + lambda1*L1 + lambda2*L2

    return cost

def gradient_descent(weight, cost, learning_rate):
    return weight - (learning_rate * T.grad(cost, weight))



def compile_model():
    X = T.matrix('X')
    Y = T.matrix('Y')

    w1, w2, b1, b2 = weights_init(input_layer_size, hidden_layer_size, 
                output_layer_size,activations = "sigmoid")

    h = predict_func(X, w1, w2, b1, b2)
    predict = theano.function([X], h)

    cost = cost_func(X, Y, h, 21, w2)

    train = theano.function(
                            [X, Y], 
                            cost, on_unused_input='ignore',
                            updates = [
                                (w1, gradient_descent(w1, cost, learning_rate)),
                                (w2, gradient_descent(w2, cost, learning_rate)),
                                (b1, gradient_descent(b1, cost, learning_rate)),
                                (b2, gradient_descent(b2, cost, learning_rate)),
                            ])

    return predict, train

predict, train = compile_model()
X, Y = get_dataset()


# Dataset error checking:
if np.shape(X)[0] == np.shape(Y)[0]:
    D = np.shape(X)[0]
else:
    print "ERROR: DATASET IS NOT VALID! Number of examples does not match number of labels."

# initialize progressbar class
progress = progressbar(iterations)
cost_list = []
for i in range(iterations):
    cost = train(X,Y)
    cost_list.append(cost)
    progress.show(i)


# Plot the cost
plt.plot(cost_list)
pylab.show()

predictions = predict(X)
for x in range(np.shape(Y)[0]):
    print "Actual value: ", Y[x], "| predicted value: ",  predictions[x]



# Give user ablility to manualy test his network
user_input = raw_input("Do you want to test your network now? (y/n): ")
if user_input == 'y':
    print "Use ctrl+c to exit"
    try:
        while(1):
            temp_values = []
            for i in range(input_layer_size):
                temp_values.append(raw_input("Enter value number %d: "%i))

            values = []
            for val in temp_values:
                val = int(val)
                values.append(val) 

            x = np.asarray([values])
            print predict(x)

    except KeyboardInterrupt:
        print "\nProgram closed"

