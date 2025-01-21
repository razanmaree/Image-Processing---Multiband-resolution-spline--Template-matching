import cv2
import numpy as np
import matplotlib.pyplot as plt


levels=6


def generate_gaussian_pyramid(image, levels, resize_ratio):

    #image = np.float32(image)
    pyramid = [image]#the image in the first level is the original image
    for _ in range(levels - 1):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        resized_image = cv2.resize(blurred, None, fx=resize_ratio, fy=resize_ratio)
        pyramid.append(np.float32(resized_image))
        image = resized_image
    return pyramid




def get_laplacian_pyramid(image, levels, resize_ratio=0.5):

    # Generate the Gaussian pyramid
    gaussian_pyramid = generate_gaussian_pyramid(image, levels, resize_ratio)
    
    # Initialize the Laplacian pyramid list
    laplacian_pyramid = []

    # Iterate through each level of the Gaussian pyramid
    for i in range(levels - 1):
        upsampled = cv2.resize(gaussian_pyramid[i + 1], (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = gaussian_pyramid[i] - upsampled
        
        # Append the Laplacian to the Laplacian pyramid
        laplacian_pyramid.append(laplacian)
    
    # Append the last level of the Gaussian pyramid to the Laplacian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid





def restore_from_pyramid(pyramidList, resize_ratio=2):
    # Start from the last/smallest level of the Laplacian pyramid
    image = pyramidList[-1]

    # Iterate over the levels of the pyramid in reverse order
    for level in reversed(pyramidList[:-1]):
        # Upsample the image to match the size of the current level
        image = cv2.resize(image, (level.shape[1], level.shape[0]))

        # Add the upsampled level to the current level to reconstruct the image
        image = level + image
       
    return image


def validate_operation(img):
    pyr = get_laplacian_pyramid(img, levels)
    img_restored = restore_from_pyramid(pyr)

    plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
    plt.imshow(img_restored, cmap='gray')

    plt.show()




def blend_pyramids(levels):
    blended_pyramid = []
    num_levels = len(levels)
    
    for i, level in enumerate(levels):
        # Calculate the index for the opposite end of the pyramid
        opposite_i = num_levels - 1 - i
        
        # Get the images from the current level
        orange_image = level[0]
        apple_image = level[1]


        #Define a mask in the size of the current pyramid.
        rows, cols = orange_image.shape
        mask = np.zeros((rows, cols), dtype=np.float32)


        #Initialize the mask’s columns, from the first one up to (0.5 * width - curr_level) to 1.0
        for col in range(int(0.5 * cols - (opposite_i))):
            mask[:, col] = 1.0

        #For each column i in the range of 0.5 * width + curr_level((, set the value to 0.9 - 0.9 * i / (2 * curr_level).(i rewrited it according to match my function)
        for col in range(int(0.5 * cols - (opposite_i)), int(0.5 * cols + (opposite_i))):
            mask[:, col] = 0.9 - 0.9 * (col - (0.5 * cols - (opposite_i))) / (2 * (opposite_i))


        #Finally, the blended pyramid level for curr_level is given by: orange * mask + apple * (1 - mask)
        blended_level = orange_image * mask + apple_image * (1 - mask)
        blended_pyramid.append(blended_level)
    
    return blended_pyramid



apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)


# validate_operation(apple)
# validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple,levels, resize_ratio=0.5)
pyr_orange = get_laplacian_pyramid(orange , levels, resize_ratio=0.5)


pyr_result = []

# Your code goes here
for i in range (len(pyr_apple)):
	pyr_result.append([pyr_orange[i],pyr_apple[i]])
pyr_result = blend_pyramids(pyr_result)



final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)

