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

def convolve_color(layers, kernal):
    convolved_image = []
    for p in layers:
        convolved_image.append(convolve(p, kernal))
    return convolved_image

def ReLU(x):
    if np.where(x>0):
        return x
    return 0

def max_pool(img, region):
    
    down_sample = np.zeros([int(img.shape[0]/region[0]), int(img.shape[0]/region[1])])
    for x in range(0, img.shape[0], region[0]):
        for y in range(0, img.shape[1], region[1]):
            max_val = np.max(img[x:x+region[0], y:y+region[1]])
            xcord = int(x/region[0])
            ycord = int(y/region[1])
            down_sample[xcord][ycord] = max_val
    
    return down_sample

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

#Structure
#=========
#num_layers = 3
#input_image_size = [400,400]
#input_layer = np.zeros(input_image_size)
#kernal_size = (3,3)
#num_kernals = 3
#weight = [np.random.randn([kernal_size, num_kernals]) for i in num_layers-1]


#Test Code
#=========
image = cv2.imread('test_image.jpg')
image = image[0:400, 0:400] #This is a very crude solution. Will fix dimension problem later

b,r,g = cv2.split(image)
new_im = max_pool_color((b,r,g), [5,5])
plt.imshow(new_im)
plt.show()