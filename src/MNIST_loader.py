import gzip
import pickle
import numpy as np

def load_data():
	f = gzip.open("../data/mnist.pkl.gz")
	f.seek(0) #Just a solution from StackExchange
	training_data, validation_data, test_data = pickle.open(f, encoding='latin1')
	f.close()
	return (training_data, validation_data, test_data)

def wrapper():
	

def vectorized_result(i):
	v_result = np.zeros([10,1])
	v_result[i] = 1.0
	return v_result
