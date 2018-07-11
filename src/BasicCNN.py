import numpy as np
import cv2
import matplotlib.pyplot as plt

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
            #print(padded_kernal) Uncomment for fun
            
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

def max_pool(img, region):
    down_sample = np.zeros([int(img.shape[0]/region[0]), int(img.shape[0]/region[1])])
    for x in range(0, img.shape[0], region[0]):
        for y in range(0, img.shape[1], region[1]):
            #print(x, y)
            max_val = np.max(img[x:x+region[0], y:y+region[1]])
            xcord = int(x/region[0])
            ycord = int(y/region[1])
            down_sample[xcord][ycord] = max_val
    return down_sample

image = cv2.imread('opencv_logo.png')
b,r,g = cv2.split(image)
#print(g[500, 500])
region = [100,100]
print(int(200/region[0]))
print(max_pool(g, region))


#kernal = np.eye(50)
#resolution_map = convolve(b, kernal)
#transform = ReLU(resolution_map)
#plt.imshow(transform)
plt.show()