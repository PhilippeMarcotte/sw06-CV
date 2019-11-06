import cv2
import numpy as np
import rospy
import math
import sys
import time
from numpy.lib.stride_tricks import as_strided

## Software Exercise 6: Choose your category (1 or 2) and replace the cv2 code by your own!

## CATEGORY 1
# def inRange(hsv_image, low_range, high_range):
# 	output = np.ones(hsv_image.shape[:2], dtype=np.uint8)
# 	for i in range(hsv_image.shape[-1]):
# 		output &= hsv_image[:,:,i] > low_range[i]
# 		output &= hsv_image[:,:,i] < high_range[i]

	
# 	cv2_output = cv2.inRange(hsv_image, low_range, high_range)
# 	if np.allclose(output, cv2_output):
# 		print("basic_output")
# 		return output
# 	else:
# 		print("cv2_output")
# 		return cv2_output

def inRange(hsv_image, low_range, high_range):
	output = np.ones(hsv_image.shape[:2], dtype=np.uint8)
	for i in range(hsv_image.shape[-1]):
		output &= hsv_image[:,:,i] > low_range[i]
		output &= hsv_image[:,:,i] < high_range[i]

	return output

# def bitwise_or(bitwise1, bitwise2):
# 	basic_or = bitwise1 | bitwise2
# 	numpy_or = np.bitwise_or(bitwise1, bitwise2)
# 	cv2_or = cv2.bitwise_or(bitwise1, bitwise2)

# 	if np.allclose(basic_or, cv2_or):
# 		print("basic_or")
# 		return basic_or
# 	elif np.allclose(numpy_or, cv2_or):
# 		print("numpy_or")
# 		return numpy_or
# 	else:
# 		print("cv2_or")
# 		return cv2_or
	
def bitwise_or(bitwise1, bitwise2):
	return bitwise1 | bitwise2

def bitwise_and(bitwise1, bitwise2):
	return bitwise1 & bitwise2

# def getStructuringElement(shape, size):
# 	if shape == cv2.MORPH_ELLIPSE:
# 		a = size[1] / 2
# 		b = size[0] / 2

# 		element = np.zeros((size[1], size[0]), dtype=np.uint8)
# 		for i in range(size[1]):
# 			dy = i - a
# 			dx = int(round(b * np.sqrt(1 - dy ** 2 / a ** 2)))
# 			ellispe_left_side = max(b - dx, 0)
# 			ellipse_right_side = min(b + dx + 1, size[0])
		
# 			element[i, ellispe_left_side:ellipse_right_side] = 1

# 		cv2_element = cv2.getStructuringElement(shape, size)
# 		if np.allclose(element, cv2_element):
# 			rospy.loginfo("basic_element")
# 			return element
# 		else:
# 			rospy.loginfo("cv2_element")
# 			return cv2_element

def getStructuringElement(shape, size):
	if shape == cv2.MORPH_ELLIPSE:
		a = size[1] / 2
		b = size[0] / 2

		element = np.zeros((size[1], size[0]), dtype=np.uint8)
		for i in range(size[1]):
			dy = i - a
			dx = int(round(b * np.sqrt(1 - dy ** 2 / a ** 2)))
			ellispe_left_side = max(b - dx, 0)
			ellipse_right_side = min(b + dx + 1, size[0])
		
			element[i, ellispe_left_side:ellipse_right_side] = 1

		return element

# def dilate(bitwise, kernel):
# 	kernel = np.flipud(np.fliplr(kernel))

# 	# We assume that the convolution is always of the same type and that the kernel is square,
# 	# so the padding equation is always the same
# 	padding = kernel.shape[0] - 1

# 	bitwise_padded = np.pad(bitwise, padding, mode='constant')

# 	output_shape = ((bitwise_padded.shape[0] - kernel.shape[0])//1 + 1,
#                     (bitwise_padded.shape[1] - kernel.shape[0])//1 + 1)
# 	output_w = as_strided(bitwise_padded, shape = output_shape + kernel.shape, 
#                         strides = (bitwise_padded.strides[0],
#                                    bitwise_padded.strides[1]) + bitwise_padded.strides)
	
# 	output_w = output_w.reshape(-1, *kernel.shape)
# 	output = (output_w * kernel).max(axis=(1,2)).reshape(output_shape)
# 	diff = int(kernel.shape[0]/2)
# 	output = output[diff:-diff, diff:-diff]

# 	cv2_output = cv2.dilate(bitwise, kernel)
# 	if np.allclose(output, cv2_output):
# 		print("basic_output")
#  		return output
# 	else:
# 		print("cv2_output")
# 		return cv2_output

def dilate(bitwise, kernel):
	kernel = np.flipud(np.fliplr(kernel))

	# We assume that the convolution is always of the same type and that the kernel is square,
	# so the padding equation is always the same
	padding = kernel.shape[0] - 1

	bitwise_padded = np.pad(bitwise, padding, mode='constant')

	output_shape = ((bitwise_padded.shape[0] - kernel.shape[0])//1 + 1,
                    (bitwise_padded.shape[1] - kernel.shape[0])//1 + 1)
	output_w = as_strided(bitwise_padded, shape = output_shape + kernel.shape, 
                        strides = (bitwise_padded.strides[0],
                                   bitwise_padded.strides[1]) + bitwise_padded.strides)
	
	output_w = output_w.reshape(-1, *kernel.shape)
	output = (output_w * kernel).max(axis=(1,2)).reshape(output_shape)
	diff = int(kernel.shape[0]/2)
	output = output[diff:-diff, diff:-diff]

	return output


## CATEGORY 2
def Canny(image, threshold1, threshold2, apertureSize=3):
	return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


## CATEGORY 3 (This is a bonus!)
def HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
	return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)