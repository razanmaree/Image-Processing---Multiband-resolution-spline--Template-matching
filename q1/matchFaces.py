import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings
warnings.filterwarnings("ignore")


def scale_down(image, resize_ratio):
    # Apply Fourier transform to the image
    f_image = fft2(image)
    
    # Shift the zero frequency component to the center
    f_image_shifted = fftshift(f_image)
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Calculate the center of the image
    crow, ccol = rows // 2, cols // 2
    
    # Calculate the new radius for the scaled image
    new_radius = int(min(rows, cols) * resize_ratio)
    
    # Calculate the distance from each pixel to the center
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    distance = np.sqrt(x**2 + y**2)
    
    # Define the mask based on the resize_ratio
    mask = np.where(distance <= new_radius, 1, 0)
    
    # Apply the mask to the shifted Fourier transform
    f_scaled_image = f_image_shifted * mask
    
    # Shift the zero frequency component back to the corner
    f_scaled_image_shifted = ifftshift(f_scaled_image)
    
    # Compute the inverse Fourier transform
    scaled_image = np.abs(ifft2(f_scaled_image_shifted))
    
    # Normalize the image to 0-255 range and convert to uint8 format
    scaled_image = np.uint8(255 * (scaled_image - np.min(scaled_image)) / np.ptp(scaled_image))
    
    # Resize the scaled image to the desired size
    scaled_height = int(image.shape[0] * resize_ratio)
    scaled_width = int(image.shape[1] * resize_ratio)
    scaled_image = cv2.resize(scaled_image, (scaled_width, scaled_height))
    
    return scaled_image


def scale_up(image, resize_ratio):
    tmp = resize_ratio
    resize_ratio = 1/( 0.5*(resize_ratio - 1))

    fourier_transform = fft2(image)
    shifted_fourier_transform = fftshift(fourier_transform)

    # Calculate padded image to calculate the doubled image
    to_pad_image = shifted_fourier_transform
    x1 = x2 = int(to_pad_image.shape[0] // resize_ratio)
    y1 = y2 = int(to_pad_image.shape[1] // resize_ratio)
    
    if (to_pad_image.shape[0] % 2 != 0) and resize_ratio %2 ==0:
        x1 = int(to_pad_image.shape[0] // resize_ratio)+1
        
    if to_pad_image.shape[1] % 2 != 0 and resize_ratio %2 ==0:
        y1 = int(to_pad_image.shape[1] // resize_ratio)+1

    padded_image = np.pad(to_pad_image, ((x1, x2), (y1, y2)), mode='constant')

    # Inverse transform
    shifted_padded_fourier_transform = fftshift(padded_image)
    shifted_padded_fourier_transform = fft2(shifted_padded_fourier_transform)
    padded_fourier_spectrum =  np.abs(shifted_padded_fourier_transform)

    flipped_img =  np.fliplr(padded_fourier_spectrum)
    flipped_img =  np.flipud(flipped_img)

    # Brightness
    flipped_img = (flipped_img - flipped_img.min()) / (flipped_img.max() - flipped_img.min()) * 255
    brightened_img = flipped_img + 70

    # Ensure pixel values remain in the range [0, 255]
    flipped_img = np.clip(brightened_img, 0, 255)
    
    return flipped_img


from numpy.lib.stride_tricks import sliding_window_view
def ncc_2d(image, pattern):
    # Create sliding windows from the image
    windows = sliding_window_view(image, pattern.shape)

    # Reshape pattern for broadcasting
    pattern_reshaped = pattern.reshape(1, *pattern.shape)

    # Calculate mean of image and pattern
    mean_image = np.mean(windows, axis=(-2, -1))
    mean_pattern = np.mean(pattern)

    # Subtract mean from image and pattern
    image_minus_mean = windows - mean_image[..., np.newaxis, np.newaxis]
    pattern_minus_mean = pattern_reshaped - mean_pattern

    # Calculate sum of squares for image and pattern
    ss_image = np.sum(image_minus_mean ** 2, axis=(-2, -1))
    ss_pattern = np.sum(pattern_minus_mean ** 2)

    # Calculate cross-correlation
    cross_corr = np.sum(image_minus_mean * pattern_minus_mean, axis=(-2, -1))

    # Calculate denominator
    denominator = np.sqrt(ss_image * ss_pattern)

    # Avoid division by zero
    denominator[denominator == 0] = 1  # Set denominator to 1 where it's zero to avoid division by zero

    # Calculate normalized cross-correlation image
    ncc = cross_corr / denominator

    return ncc


def display(image, pattern):
	
	plt.subplot(2, 3, 1)
	plt.title('Image')
	plt.imshow(image, cmap='gray')
		
	plt.subplot(2, 3, 3)
	plt.title('Pattern')
	plt.imshow(pattern, cmap='gray', aspect='equal')
	
	ncc = ncc_2d(image, pattern)
	
	plt.subplot(2, 3, 5)
	plt.title('Normalized Cross-Correlation Heatmap')
	plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto') 
	
	cbar = plt.colorbar()
	cbar.set_label('NCC Values')
		
	plt.show()


def draw_matches(image_gray, matches, pattern_size):
    # Convert grayscale image back to BGR
    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    
    last_rect = None
    
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        
        current_rect = (top_left, bottom_right)
        
        # Check if current rectangle intersects with the last drawn rectangle
        if last_rect is not None and rectangles_intersect(current_rect, last_rect):
            continue  # Skip drawing this rectangle
        
        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
        last_rect = current_rect

    cv2.imshow('result', image)
    #cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)

    
def rectangles_intersect(rect1, rect2):
    """
    Check if two rectangles intersect or not.
    rect1 and rect2 are tuples of ((x1, y1), (x2, y2)) representing top-left and bottom-right corners.
    """
    x1_intersect = max(rect1[0][0], rect2[0][0]) <= min(rect1[1][0], rect2[1][0])
    y1_intersect = max(rect1[0][1], rect2[0][1]) <= min(rect1[1][1], rect2[1][1])
    return x1_intersect and y1_intersect

#------------------------------------------------------------------------------------------------------------

CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)


'''
############# Students #############
resize_ratio = 0.45

patten_scaled = scale_down(pattern, resize_ratio)                        

display(image, patten_scaled)

ncc = ncc_2d(image, patten_scaled) 

# Thresholding to find good matches
threshold = 0.51
real_matches = np.argwhere(ncc > threshold)

real_matches[:,0] += patten_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += patten_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, patten_scaled.shape)	# if pattern was not scaled, replace this with "pattern"
'''



############# Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

resize_ratio = 0.23
patten_scaled = scale_down(pattern, resize_ratio) 

display(image, patten_scaled)

ncc = ncc_2d(image, patten_scaled)          

threshold = 0.3985
real_matches = np.argwhere(ncc > threshold)


######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += patten_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += patten_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, patten_scaled.shape)	# if pattern was not scaled, replace this with "pattern"









