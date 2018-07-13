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

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
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
        for epoch in range(epochs):
            for batch in mini_batches:
                self.update_weight_bias(batch, learning_rate)
            if test_data:
                self.evaluate(test_data)
       
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
        for i in range(2, self.num_layers):
            temp = np.dot(self.weight[-i+1].transpose(), error[-i+1])
            sp = sigmoid_prime(self.z_layer[-i])
            error[-i] = np.multiply(temp, sp)

	    #Equation 3
        delta_bias = error

	    #Equation 4
        delta_weight = []
        for i in range(2, self.num_layers):
            delta_weight.append(np.dot(error[-i], self.layer[-i-1].transpose()))

        return delta_weight, delta_bias

    def update_weight_bias(self, mini_batch, learning_rate):
        nabla_weight = [np.zeros(i.shape) for i in self.weight]
        nabla_bias = [np.zeros(i.shape) for i in self.bias]

        for i in range(len(mini_batch)):
             y = mini_batch[i][1]#Expected Output
             x = mini_batch[i][0]#Input
             self.layer[0] = x
             self.feedforward()#Propogate through the network for output
             delta_weight, delta_bias = self.backprop(y)
             
             nabla_weight = [np.add(nabla_weight[i],delta_weight[i]) for i in range(len(self.weight))]
             nabla_bias = [np.add(nabla_bias[i],delta_bias[i]) for i in range(len(self.bias))]

        for i in range(len(self.weight)): 
             self.weight[i] = self.weight[i]-((learning_rate/len(mini_batch))*nabla_weight[i])
             self.bias[i] = self.bias[i]-((learning_rate/len(mini_batch))*nabla_bias[i])

    def evaluate(self, test_data):
        correct = 0
        for i in test_data:
            self.layer[0] = i[0]
            if (np.argmax(self.layer[-1])==i[1]):
                correct+=1
        print("Percentage correct: "+str(correct/len(test_data)))

#Global Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

net = NeuralNet([784,100,10])
training_data, validation_data, test_data = MNIST_loader.wrapper()
net.SGD(training_data, 30, 20, 3.0, test_data=test_data)
print("DONE!!!")
net.evaluate(test_data)
print(net.layer[-1])
