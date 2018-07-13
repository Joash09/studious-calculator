import numpy as np

class NeuralNet():

    def __init__(self, structure):
        self.num_layer = len(structure)
        self.layers = [np.zeros([i, 1]) for i in structure]
        self.bias = self.layers
        self.weights = [np.zeros([structure[i], structure[i-1]]) for i in range(1, self.num_layer)]

    def feedforward(self):
        for i in range(1, self.num_layers):
            self.layer[i] = softmax(np.dot(weight[i-1], layer[i-1])+bias[i-1])
    
    def backprop(self, x, y):
        error = [np.zeros(layers[i].shape) for i in range(1, self.num_layers)]
        delta_weight = [np.zeros(w.shape) for w in self.weights]
        delta_bias = [np.zeros(b.shape) for b in self.bias]
        
        #Equation 1
        error[-1] = layers[-1]-y

        #Equation 3
        delta_bias[-1] = error[-1]

        #Equation 4
        delta_weight[-1] = np.dot(self.layer[-1], error[-1])

#Minimizes the log likelihood function
def softmax(x):
    result = [(i/np.sum(x)) for i in x]
    return result
