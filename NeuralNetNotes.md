# Structure of the network
* Rows in the matrix will correspond to a neuron in the that specific layer.

* The structure of the weight matrix is such that the number of rows correspond to the number of neurons for the upcoming layer and the number of columns correspond to the number of neurons in the previous layer. 

# Feedforward notes
* The process of moving the information from one layer to the next. 

* The weight of a neuron tells us how important that neuron is important to the system
* The bias tells us how easily the neuron will fire (turn on). 

* The purpose of the sigmoid function is to squash the ouput to be between the bounds of 0 and 1 which is easier for us to use. That is to say small changes in the weights and bias won't result strong on or strong off and that we can gage what those changes in the weights and biases are doing.

# Backpropogation
* Supervised learning means we have the answer for the training data. (i.e. we can calculate the error)
* The error we calculate at the output is the result of errors at EACH individual layer (i.e. e(H1) = a*(e(Y1)) +b*e(Y2))


* Find the change each weight and bias makes to the cost function. The chain rule comes into play was the weights and biases from one layer is affected by the weights and biases from the previous layers.
* The change for each element in each layer will will be calculated to give us the gradient vector which will tell how the weights and biases will need to change such that we can minimize the cost function.         
