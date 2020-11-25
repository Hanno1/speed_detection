import cv2
import numpy as np


VALUE = 10
MARGIN = 0
PERIMETER = 300


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


def get_contours_1(image):
    # lower and upper bounding in bgr format for the color of the shield
    lower = np.array([0, 0, 50])
    upper = np.array([120, 80, 255])
    # creates an mask which filters through the image and sets values in the bounding box to 1 (else 0)
    mask = cv2.inRange(image, lower, upper)

    # now we get an colored image with the bitwise_and function
    image_result = cv2.bitwise_and(image, image, mask=mask)
    """stacked_result = stack_images(SIZE, [[image, mask, image_result]])
    cv2.imshow("First transformation", stacked_result)
    cv2.waitKey(0)"""
    get_contours_2(image_result, image)


def get_contours_2(image, original):
    """
    takes an image already edited with the bgr bounding, transforms it into an hsv and use an hsv bounding """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # bounding box
    lower = np.array([160, 0, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(image_hsv, lower, upper)
    """stacked_result = stack_images(SIZE, [[image, image_hsv, mask]])
    cv2.imshow("Result", stacked_result)
    cv2.waitKey(0)"""
    # mask = cv2.bitwise_not(mask)
    # creates an rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # erodes and then dilates the mask with the kernel
    mask = cv2.erode(mask, kernel=kernel, iterations=1)
    mask = cv2.dilate(mask, kernel=kernel, iterations=1)
    # smooth the image with a gaussian filter
    image_blur = cv2.GaussianBlur(mask, (5, 5), 1)
    """stacked_result = stack_images(SIZE, [[image_hsv, mask, image_blur]])
    cv2.imshow("Second Transformation", stacked_result)
    cv2.waitKey(0)"""
    get_contours_3(image_blur, original)


def get_contours_3(image, original):
    """
    finally returns the contours. We take the smoothed image for contour detection """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_peri = 0
    max_area = 0
    max_vertices = 0
    image_copy = original.copy()
    x_biggest = y_biggest = w_biggest = h_biggest = 0
    # iterate through the list of contours
    for contour in contours:
        # get area, perimeter, approximation and vertices
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        vertices = len(approx)
        # get a bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        # if the contour is almost a square, the area is bigger then 100 and it has more then 4 vertices
        if w - VALUE < h < w + VALUE and PERIMETER < peri and 4 < vertices < 20\
                and 20000 < area and 100 < w:
            # we want the maximum perimeter because area is not that reliable
            if peri > max_peri:
                # save the coordinates
                x_biggest = x
                y_biggest = y
                w_biggest = w
                h_biggest = h
                # revalue the maximum perimeter
                max_peri = peri
                max_area = area
                max_vertices = vertices
            # draws an rectangle that satisfies the entire if conditions
            cv2.rectangle(image_copy, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), (0, 0, 200), 20)
    cv2.rectangle(image_copy, (x_biggest - MARGIN, y_biggest - MARGIN),
                              (x_biggest + w_biggest + MARGIN, y_biggest + h_biggest + MARGIN),
                              (0, 255, 0), 40)
    print(h_biggest, w_biggest)
    print(max_vertices, max_area, max_peri)
    stacked_result = stack_images(SIZE, [[original, image, image_copy]])
    cv2.imshow("finally!", stacked_result)
    cv2.waitKey(0)


# define the size
SIZE = 0.1
img = cv2.imread("../milestone1/pictures/50er3.jpg")
get_contours_1(img)
