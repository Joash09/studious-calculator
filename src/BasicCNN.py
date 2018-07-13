import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    down_sample = np.zeros([11,11])
    #down_sample = np.zeros([int(img.shape[0]/region[0]), int(img.shape[1]/region[1])])
    for x in range(0, img.shape[0], region[0]):
        for y in range(0, img.shape[1], region[1]):
            max_val = np.average(img[x:x+region[0], y:y+region[1]])
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

#Different types of filters
#==========================
identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
edge_detection1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
edge_detection2 = np.array([[-1,-1,-1], [-1,-8,-1], [-1,-1,-1]])#Rubbish
edge_detection3 = np.array([[1,0,-1], [0,0,0], [-1,0,1]]) #Works really well
sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
guassion_blur = np.array([[1,2,1], [2,4,2],[1,2,1]])*(1/16)


#Test Code
#=========
image = cv2.imread('test2.jpg', 0)
#plt.show()
#plt.plot()

plt.figure(1)
#Edge dectection 1
resolution_1 = convolve(image, edge_detection1, 1)
plt.subplot(211)
plt.imshow(resolution_1, cmap='gray')

#Edge dectection 3
resolution_2 = convolve(image, edge_detection3, 1)
plt.subplot(212)
plt.imshow(resolution_2, cmap='gray')
plt.show()

#Max Pool
#resolution_2 = convolve(resolution_2,guassion_blur,1)
#resolution_2 = np.fft(resolution_2)
print(resolution_2)
plt.figure(2)
plt.imshow(resolution_2)
plt.show()

print(image_to_input(resolution_2))