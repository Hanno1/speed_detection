import cv2
import numpy as np
import math


def stack_images(image_scale, image_array):
    """
    takes an positive real number and an matrix containing images.
    arrange them in the given order and scales them up (scale > 1)
    or down (scale < 1)
    """
    # rearranges images with 1 dimension (gray images)
    for r in range(0, len(image_array)):
        for i in range(0, len(image_array[0])):
            # now we consider every image separately and resize it
            image_array[r][i] = cv2.resize(image_array[r][i], (int(image_array[r][i].shape[1] * image_scale),
                                                               int(image_array[r][i].shape[0] * image_scale)))
            # if the shape is only two dimensional we have to make it three dimensional
            if len(image_array[r][i].shape) == 2:
                image_array[r][i] = cv2.cvtColor(image_array[r][i], cv2.COLOR_GRAY2BGR)
            elif len(image_array[r][i].shape) != 3:
                raise ValueError
            else:
                pass
    # horizontal stacks every image in the first row and saves it as first_image_row
    first_image_row = image_array[0][0]
    for i in range(1, len(image_array[0])):
        first_image_row = np.hstack((first_image_row, image_array[0][i]))
    # now we go through every row of the matrix and create a horizontal stacked image_row
    # then we concatenate all rows vertical to the first_image_row
    for row in range(1, len(image_array)):
        image_row = image_array[row][0]
        for i in range(1, len(image_array[row])):
            image_row = np.hstack((image_row, image_array[row][i]))
        first_image_row = np.vstack((first_image_row, image_row))
    return first_image_row


def get_contours_bgr(image):
    """
    takes an image and computes all points in range of a given color bonding box (BGR) """
    # lower and upper bounding in bgr format for the color of the shield
    lower = np.array([120, 50, 0])
    upper = np.array([190, 110, 115])
    # creates an mask which filters through the image and sets values in the bounding box to 1 (else 0)
    mask = cv2.inRange(image, lower, upper)

    # now we get an colored image with the bitwise_and function
    image_result = cv2.bitwise_and(image, image, mask=mask)
    get_contours_hsv(image_result, image)


def get_contours_hsv(image, original):
    """
    takes an image already edited with the bgr bounding, transforms it into an hsv and use an hsv bounding """
    # transform bgr image to hsv
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # bounding box
    lower = np.array([80, 100, 100])
    upper = np.array([120, 230, 179])
    mask = cv2.inRange(image_hsv, lower, upper)
    # creates an rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # erodes and then dilates the mask with the kernel
    mask = cv2.erode(mask, kernel=kernel, iterations=1)
    mask = cv2.dilate(mask, kernel=kernel, iterations=1)
    # smooth the image with a gaussian filter
    image_blur = cv2.GaussianBlur(mask, (5, 5), 1)
    get_contours_final(image_blur, original)


def get_contours_final(image, original):
    """
    finally returns the contours. We take the smoothed image for contour detection """
    # get the contours of the given image
    lines = cv2.HoughLines(image, 1, np.pi / 180, 150, None, 0, 0)
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(original, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    stacked_result = stack_images(SIZE, [[original, image]])
    cv2.imshow("finally!", stacked_result)
    cv2.waitKey(0)


# define the size of the outcome should be around 0.1
SIZE = 0.1
# The difference of width and height of a shield should not be bigger then Value
VALUE = 10
# Define a margin in the output image
MARGIN = 20
# define the smallest perimeter for an shield
PERIMETER = 300
# define the smallest Area
AREA = 20000

img = cv2.imread("pictures/spielstrasse1.jpg")
get_contours_bgr(img)
