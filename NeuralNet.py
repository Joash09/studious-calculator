import numpy as np
import cvxpy as cvx #Convex optimization package

# We will still learn how to read into the input vectors 
# (rows, columns)

class NeuralNet(x1, x2, x3):
    def __init__():
        #Strucutre
        size_input = x1
        size_hidden = x2
        size_output = x3
        
        #layers as column vectors
        input_layer = np.zeros([size_input,1])
        hidden_layer = np.zeros([size_hidden, 1])
        output_layer = np.zeros([size_output, 1])

        #Weights and biases
        weight1 = np.random.randn([size_hidden, size_input])
        bias1 = np.random.randn([size_hidden, 1])

        weight2 = np.random.randn([size_output, size_hidden])
        bias2 = np.random.randn([size_output, 1])

    def sigmoid(x):
        return 1/(1+np.e**(-x))

    def forward_prop(layer, prev_layer, weights, bias):
        layer = sigmoid(np.dot(weights, prev_layer)+bias)

    def backprop():
        
    
    def MSF(output, expected):
        """Mean squares cost fucntion"""
        return (output-expected)**2

 class Application():
    def __main__():
        net = new NeuralNet(100,15,10)
        
        #We want to feedforward, 
        #Backpropogation!!!
        #work out the cost of output(error), adjust weights and bias in relation to the error
        #Input, feedforward, output error, backprop, gradient

        
