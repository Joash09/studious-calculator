import numpy as np
from random import shuffle
import MNIST_loader

class NeuralNet:

    def __init__(self, structure):

        self.num_layers = len(structure)
        self.layer = []; self.z_layer = []; self.weight=[]; self.bias=[];
        
        for i in range(self.num_layers):
            self.layer.append(np.zeros([structure[i], 1]))
            self.z_layer.append(np.zeros([structure[i], 1]))
           
        for i in range(1, self.num_layers):
            self.weight.append(np.random.randn(structure[i], structure[i-1]))
            self.bias.append(np.random.randn(structure[i], 1))

    def feedforward(self):
        for i in range(1, self.num_layers):
            self.z_layer[i] = np.dot(self.weight[i-1], self.layer[i-1])+self.bias[i-1]
            self.layer[i] = sigmoid(self.z_layer[i])

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate):
        n = len(training_data)
        
	#Set the mini batches
        mini_batches = []
        for i in range(0, n, mini_batch_size):
            mini_batches.append(training_data[i:i+mini_batch_size])
        shuffle(mini_batches)

	#Iterate through each mini batch
	#Iterate through each training element
	#Backprop
	#Adjust weights
        for i in range(mini_batch_size):
             y = mini_batches[0][i][1]#Expected Output
             x = mini_batches[0][i][0]#Input
             self.layer[0] = x
             self.feedforward()#Propogate through the network for output
             delta_weight, delta_bias = self.backprop(y)

    def backprop(self, y):
        def cost_derivative(y):
            return self.layer[-1]-y
        
        error = []
        for i in range(1, self.num_layers):
            error.append(np.zeros(self.layer[i].shape))

	#Equation 1
        sp = sigmoid_prime(self.z_layer[-1])
        error[-1] = np.multiply(cost_derivative(y), sp)

	#Equation 2
        for i in range(self.num_layers-2, 0, -1):
            temp = np.dot(self.weight[i].transpose(), error[i])
            sp = sigmoid_prime(self.z_layer[i])
            error[i] = np.multiply(temp, sp)

	#Equation 3
        delta_bias = error

	#Equation 4
        delta_weight = []
        for i in range(len(self.weight)):
            delta_weight.append(np.dot(error[i], self.layer[i+1].transpose()))
            print(delta_weight[i])

        return delta_weight, delta_bias

    def update_weight_bias(self, learning_rate):
        pass 

#Global Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

net = NeuralNet([784,30,10])
training_data, validation_data, test_data = MNIST_loader.wrapper()
net.SGD(training_data, 100, 20, 0.1)
