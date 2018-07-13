import numpy as np
import cv2
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet

def convolve(image, kernal, stride):
    #Still need to add zero_padding functionality
    padded_kernal = np.zeros(image.shape)
    padding = np.array(image.shape)-np.array(kernal.shape)
    resolution_map = np.zeros(padding)
    print(resolution_map.shape)
    for x_shift in range(0, padding[0], stride):
        for y_shift in range(0, padding[1], stride):
            #Shifting the kernal
            for x_k in range(kernal.shape[0]):
                for y_k in range(kernal.shape[1]):
                    padded_kernal[x_shift+x_k][y_shift+y_k]=kernal[x_k][y_k]
            #print(padded_kernal) Uncomment for fun
            
            #Do convolution
            temp = np.sum(np.multiply(image, padded_kernal))
            resolution_map[x_shift][y_shift] = temp
            #Reset the padded kernal
            padded_kernal = np.zeros(image.shape)
    
    return resolution_map.astype(np.uint8)

def convolve_color(image, kernal, stride):
    b,g,r = cv2.split(image)
    b = convolve(b, kernal, stride)
    g = convolve(g, kernal, stride)
    r = convolve(r, kernal, stride)
    return cv2.merge((b,g,r))

def ReLU(x):
    if np.where(x>0):
        return x
    return 0

def max_pool(img, region):
    down_sample = np.zeros([int(np.floor(img.shape[0]/region[0])+1), int(np.floor(img.shape[1]/region[1])+1)])
    for x in range(0, img.shape[0], region[0]):
        for y in range(0, img.shape[1], region[1]):
            max_val = np.max(img[x:x+region[0], y:y+region[1]])
            xcord = int(x/region[0])
            ycord = int(y/region[1])
            down_sample[xcord][ycord] = max_val
    return down_sample.astype(np.uint8)

def max_pool_color(img, region):
    b,g,r = cv2.split(img)
    b = max_pool(b, region)
    g = max_pool(g, region)
    r = max_pool(r, region)
    return cv2.merge((b,g,r))

def image_to_input(img):
    #Image to input column vector for the fully connected network
    #Only works for a 2D image
    w = img.shape[0]
    h = img.shape[1]
    return np.reshape((w*h), 1)

def appy_filters(img, stride):
    feature_maps = []
    
    #Different types of filters
    #==========================
    identity = np.array([[0,0,0],[0,1,0],[0,0,0]])

    edge_detection1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    edge_detection2 = np.array([[-1,-1,-1], [-1,-8,-1], [-1,-1,-1]])#Rubbish
    edge_detection3 = np.array([[1,0,-1], [0,0,0], [-1,0,1]]) #Works really well
    
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    
    guassion_blur = np.array([[1,2,1], [2,4,2],[1,2,1]])*(1/16)
    
    sobel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) #Works the best for faces
    #==========================
    
    feature_maps.append(convolve(img, edge_detection1, stride))
    #feature_maps.append(convolve(img, edge_detection2, stride))
    feature_maps.append(convolve(img, edge_detection3, stride))
    feature_maps.append(convolve(img, sobel, stride))
    #feature_maps.append(convolve(img, prewitt, stride))
    #feature_maps.append(convolve(img, gradient,stride))

    return feature_maps

#Test Code
#=========
image = cv2.imread('test3.jpg', 0)

#plt.figure(1)
#lt.imshow(image)
#plt.show()

feature_maps = appy_filters(image, 1)
#feature_maps[-1] = ReLU(feature_maps[-1])
plt.figure(3)
plt.imshow(max_pool(feature_maps[-1], [5,5]), cmap='gray')
plt.show()

plt.figure(4)
plt.imshow(max_pool(feature_maps[-1], [5,5]), cmap='gray')
plt.show()

plt.figure(2)
for i in range(1,len(feature_maps)+1):
    plt.subplot(str(len(feature_maps))+str(1)+str(i))
    plt.imshow(ReLU(feature_maps[i-1]), cmap='gray')
#plt.show()

input_layer = image_to_input(feature_maps[0])
#print(image_to_input(feature_maps[0]))
net = NeuralNet([9409, 20, 10])
net.layers[0] = input_layer
print(net.layers[0])