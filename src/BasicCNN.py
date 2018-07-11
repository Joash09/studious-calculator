import numpy as np

def convolve(image, kernal):
    padded_kernal = np.zeros(image.shape)
    padding = np.array(image.shape)-np.array(kernal.shape)
    resolution_map = np.zeros(padding+1)

    for x_shift in range(padding[0]+1):
        for y_shift in range(padding[1]+1):
            #Shifting the kernal
            for x_k in range(kernal.shape[0]):
                for y_k in range(kernal.shape[1]):
                    padded_kernal[x_shift+x_k][y_shift+y_k]=kernal[x_k][y_k]
            print(padded_kernal)
            
            #Do convolution
            temp = np.sum(np.multiply(image, padded_kernal))
            resolution_map[x_shift][y_shift] = temp
            
            #Reset the padded kernal
            padded_kernal = np.zeros(image.shape)
    return resolution_map

def ReLU(x):
    if np.where(x>0):
        return x
    return 0

image = np.random.randint(0, 10, size=(3,9))
kernal = np.ones([2,2])
print(image); print(kernal)
resolution_map = convolve(image, kernal)
transform = ReLU(resolution_map)
print(transform)