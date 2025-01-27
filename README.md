# **Image-Processing---Multiband-resolution-spline--Template-matching**
## ***Question 1 – Template matching:***
We want to detect all the faces within a given image and we are going to do that for two different images, with the same face template. To get started on that, do the following in the python script file ‘matchFaces.py’:
1. Implement the function 'scale_down(image, resize_ratio)'.
Given an image, use a Fourier transform to scale down the image (optimal interpolation). Leave ratio*size from the original image(Also, consider possibly blurring the image before scaling it down).

2. Implement the function 'scale_up(image, resize_ratio)'.
Given an image, use a Fourier transform to scale up the image. The output should be ratio*size from the original image. You may assume the ratio is at least 1.

3. Implement the function 'ncc_2d(image, pattern)'.
Given an image and a pattern image, returns the NCC image.

4. The function 'display' gets an image and a pattern, and displays both, along with their NCC heatmap.

5. Using the filtered matches you found, call the function 'draw_matches' with the original image.
The result will be the same image but with red rectangles over the recognized faces.

## ***Question 2 - Multiband blending:***
Let’s go back to the blending of an apple and an orange from the lecture about multi-scale representation. Now we are going to implement it. In ‘blendFruits.py’:
1. Implement the function 'get_laplacian_pyramid'. It gets an image and returns a Laplacian pyramid with the specified levels number. Each additional level is half-size per-axis compared to the previous one.
2. Implement the function 'restore_from_pyramid'. It gets a Laplacian pyramid and returns the image.
3. Implement the function ‘blend_pyramids’.
Given an implementation guide here:
For each level (1 -> total_levels) in the pyramids:

    ● Define a mask in the size of the current pyramid.

   ● Initialize the mask’s columns (it’s basically a 2D matrix) from the first one up to (0.5 * width - curr_level) to 1.0 (some advanced numpy indexing action will be required here). We’re doing this to properly scale the blending according to the pyramid level.

     ● For each column i in the range of (0.5 * width + curr_level), set the value to 0.9 - 0.9 * i / (2 * curr_level). This is gradual blending part.

   ● Finally, the blended pyramid level for curr_level is given by:
orange * mask + apple * (1 - mask)

4. The new image is the blending between ‘orange.jpg’ and ‘apple.jpg’. Create a Laplacian Pyramid for each of the two images, blend those two pyramids per-level and then restore the blended image from the result pyramid (do all of this using the functions you’ve implemented).
