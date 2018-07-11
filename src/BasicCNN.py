import numpy as np
import cv2 as cv

class CNN:
    def ReLU(self, x):
        if (x>0):
            return x
        return 0

    def __init__(self, structure):
        self.num_layers = len(structure)

        self.layer = [np.zeros([i,1]) for i in structure]

#img = cv.imread('../data/temp.jpg')
def convolution(image, kernal):
    x_kernal = kernal.shape[0]
    y_kernal = kernal.shape[1]
    feature_map = 0
    #Iterates through each pixel
    for x in range(4):
        for y in range(4):
            kernal_pixel = 0
            for x_k in range(2):
                for y_k in range(2):
                    if (x==x_k and y==y_k):
                        kernal_pixel += img[x][y]*kernal[x_k][y_k]
            #Write to the feature map
            print(kernal_pixel)

#img = np.random.randint(0, 10, size=(4,4))
#kernal = np.ones([2,2])
#print(img)
#print(kernal)
#convolution(img, kernal)
#net = CNN([resolution,15,10])

image = cv.imread('opencv_logo.png')
filter1 = np.identity(20)
cv.convole(image, filter1)