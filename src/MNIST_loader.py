import gzip
import pickle
import numpy as np

def load_data():
	f = gzip.open("../data/mnist.pkl.gz")
	f.seek(0) #Just a solution from StackExchange
	training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
	f.close()
	return (training_data, validation_data, test_data)

def wrapper():
	tr_d, va_d, te_d = load_data()
	#Shaping training data
	training_inputs = []; training_results = [];
	for i in tr_d[0]:
		training_inputs.append(np.reshape(i, (784,1)))
	for i in tr_d[1]:
		training_results.append(vectorized_result(i))
	training_data = list(zip(training_inputs, training_results))

	#Shaping validation data
	validation_input = [];
	for i in va_d[0]:
		validation_input.append(np.reshape(i, (784,1)))
	validation_data = list(zip(validation_input, va_d[1])) #Output is already in correct format

	#Shaping test data
	test_input = []
	for i in te_d[0]:
		test_input.append(np.reshape(i, (784,1)))
	test_data = list(zip(test_input, te_d[1])) #Output is already in correct format

	return (training_data, validation_data, test_data)

def vectorized_result(i):
	v_result = np.zeros([10,1])
	v_result[i] = 1.0
	return v_result
