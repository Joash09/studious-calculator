import numpy as np

class NeuralNet:

    def __init__(self, structure):

        self.num_layers = len(structure)
        self.layer = []; self.z_layer = []; self.weight=[]; self.bias=[];
        
        for i in range(num_layers):
            layer.append(np.zeros([structure[i], 1]))
            z_layer.append(np.zeros([structure[i], 1]))
           
        for i in range(1, num_layers):
            weight.append(np.random.randn(structure[i], structure[i-1]))
            bias.append(np.random.randn(structure[i], 1)
                    
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x):
	#Derivative of activation function
	return sigmoid(x)+1/sigmoid(x)
                    
    def feedforward():
        for i in range(1, num_layers):
            z_layer[i] = np.dot(weight[i], layer[i-1])+bias[i]
        layer[i] = sigmoid(z_layer)

    def SGD(training_data, epochs, mini_batch_size, learning_rate):

        #Training data is in the form of a tuple (x,y) where x is the image and y is the reuslt
        random.shuffle(training_data)
        
	#Create batches of data
	mini_batches = []
	for i in range(0, n,  mini_batch_size):
		mini_batches.append(training_data[i:i+mini_batch_size]

        #select random training input size
        for epoch in range(epochs):
            for batch in range(mini_batches):
                layer[0] = mini_batches[batch][0]
                feedforward()
                update_mini_batch()

    def update_mini_batch():
	pass

    def backprop(x, y):
	error = []
	for i in range(1, num_layers):
		error.append(np.zeros(layer[i].shape))
	
	#Equation 1
	sp = sigmoid_prime(z_layer[-1])
	error[-1] = np.multiply(cost_derivative(), sp) #TODO Cost derivative
	#Equation 2
	for i in range(2, num_layers):
		sp = sigmoid_prime(z_layer[-1*i])
		error[-1*i] = np.multiply(np.dot(weight[i].transpose(), layer[i+1]), sp)
	#Equation 3
	dela_bias = error
	#Equation 4
	delta_weight = []
	for i in range(1, num_layers):
		delta_weight.append(np.dot(layer[i-1], error[i]))

	return delta_weight, dela_bias
