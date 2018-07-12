import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve(image, kernal, stride):
    #Still need to add zero_padding functionality
    padded_kernal = np.zeros(image.shape)
    padding = np.array(image.shape)-np.array(kernal.shape)
    resolution_map = np.zeros(padding)

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

def convolve_color(layers, kernal, stride):
    convolved_image = []
    for p in layers:
        convolved_image.append(convolve(p, kernal, stride).astype(np.uint8))
        print("Done")
    return cv2.merge((convolved_image[0], convolved_image[1], convolved_image[2]))

def ReLU(x):
    if np.where(x>0):
        return x
    return 0

def max_pool(img, region):
    
    down_sample = np.zeros([int(img.shape[0]/region[0]), int(img.shape[1]/region[1])])
    for x in range(0, img.shape[0], region[0]):
        for y in range(0, img.shape[1], region[1]):
            max_val = np.max(img[x:x+region[0], y:y+region[1]])
            xcord = int(x/region[0])
            ycord = int(y/region[1])
            down_sample[xcord][ycord] = max_val
    return down_sample.astype(np.uint8)

def max_pool_color(layers, region):
    image = []
    for layer in layers:
        image.append(max_pool(layer, region))
    return cv2.merge((image[0], image[1], image[2]))

#Different types of filters
#==========================
identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
edge_detection1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
edge_detection2 = np.array([[-1,-1,-1], [-1,-8,-1], [-1,-1,-1]])
edge_detection3 = np.array([[1,0,-1], [0,0,0], [-1,0,1]]) #Works really well
sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
guassion_blur = np.array([[1,2,1], [2,4,2],[1,2,1]])*(1/16)



#Test Code
#=========
image = cv2.imread('opencv_logo.png')
b,g,r = cv2.split(image)

new_im = max_pool_color((b,g,r), [10,10])
plt.imshow(new_im)
plt.show()

new_im = convolve(b, edge_detection3, 1)
#new_im = convolve_color((b,r,g), edge_detection3, 1)
plt.imshow(new_im)
plt.show()