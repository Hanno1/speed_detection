import numpy as np
import cv2


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


def get_bgr_values(image, lower, upper):
    """
    returns a image which has only color pixels in range of the lower, upper bgr-bounding """
    # creates an mask which filters through the image and sets values in the bounding box to 1 (else 0)
    mask = cv2.inRange(image, lower, upper)
    # now we get an colored image with the bitwise_and function
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def get_hsv_values(image, lower, upper):
    """
    returns a image which has only color pixels in range of the lower, upper hsv-bounding """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower, upper)
    # creates an rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # erodes and then dilates the mask with the kernel
    mask = cv2.erode(mask, kernel=kernel, iterations=2)
    mask = cv2.dilate(mask, kernel=kernel, iterations=2)
    # smooth the image with a gaussian filter
    image_blur = cv2.GaussianBlur(mask, (5, 5), 1)
    return image_blur


def get_contours_all_corner(image):
    """
    Part 1: takes an image and computes all points in range of a given color bonding box (BGR)
    Part 2: transforms it into an hsv and use an hsv bounding
    Part 3: returns the contours. We take the smoothed image for contour detection
    """
    # --- Part 1 --- #
    # lower and upper bounding in bgr format for the color of the shield
    lower = np.array([60, 25, 0])
    upper = np.array([190, 110, 115])
    image_result = get_bgr_values(image, lower, upper)

    # --- Part 2 --- #
    # bounding box for hsv image
    lower = np.array([80, 100, 85])
    upper = np.array([120, 230, 179])
    image_blur = get_hsv_values(image_result, lower, upper)

    # --- Part 3 --- #
    # get the contours of the given image
    contours, hierarchy = cv2.findContours(image_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # save areas which are possible shields in list
    nice_areas = []
    # iterate through the list of contours
    for contour in contours:
        # get area, perimeter, approximation and vertices
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        vertices = len(approx)
        # get a bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        """
        We want to save a area if:
        - w is approximately between h and 2h
        - perimeter is approximately 2w + 2h (like a normal rectangle)
        - the vertices are between 3 and 10 (not always recognized as a rect)
        - the height of the shield is not to small or big
        - the area is not to small
        """
        if h - VAR < w < 2*h + VAR and 2*w + 2*h - VAR < peri < 2*w + 2*h + VAR and 3 < vertices < 10\
                and img_height // 10 < h < img_height // 3 and 150000 < area:
            # save the coordinates and add them to the list
            coordinates = [x, y, w, h]
            nice_areas.append(coordinates)
    return nice_areas, image_blur


def get_contours_all_round(image):
    """
    Part 1: takes an image and computes all points in range of a given color bonding box (BGR)
    Part 2: transforms it into an hsv and use an hsv bounding
    Part 3: get the contours. We take the smoothed image for contour detection
    """
    # --- Part 1 --- #
    # lower and upper bounding in bgr format for the color of the shield
    lower = np.array([0, 0, 50])
    upper = np.array([160, 110, 255])
    image_result = get_bgr_values(image, lower, upper)

    # --- Part 2 --- #
    # bounding box for hsv image
    lower = np.array([160, 0, 0])
    upper = np.array([179, 255, 255])
    image_blur = get_hsv_values(image_result, lower, upper)

    # --- Part 3 --- #
    # get the contours of the given image
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.maxArea = 500000
    params.minArea = 10000

    params.filterByCircularity = True
    params.minCircularity = 0.7

    params.filterByConvexity = True
    params.minConvexity = 0.9

    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    detector = cv2.SimpleBlobDetector_create(params)
    keyp = detector.detect(image_blur)

    return keyp, image_blur


path = "pictures/30er5.jpg"

img = cv2.imread(path)
img_height = np.shape(img)[0]
img_width = np.shape(img)[1]
img_area = img_height * img_width

# define the size of the outcome depending on the area of the image
SIZE = img_area * 8 * 10 ** (-9)
# we only take the first two numbers after the comma
SIZE = float("{:.2f}".format(SIZE))
# The difference of width and height of a shield should not be bigger then Value.
# For my pictures its around 50 pixels
DIFF = img_area * 3 * 10 ** (-6)
# define the smallest perimeter for an shield
PERIMETER = img_area * 2 * 10 ** (-5)
# define the smallest Area depending on the area of the image
AREA = img_area * 10 ** (-3)
# Define a margin in the output image
MARGIN = 20
# Var is the difference in which the perimeter can differ
VAR = 200

# we don't need to look at pixel below normal shield height 5/6 width
img = img[0: 5 * img_height // 6, 0: img_width]
# update the height
img_height = np.shape(img)[0]

keypoints, mask1 = get_contours_all_round(img)
result_corner_areas, mask2 = get_contours_all_corner(img)

for keypoint in keypoints:
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    r = int(keypoint.size)
    cv2.circle(img, (x, y), r, (0, 0, 255), 20)

for rect in result_corner_areas:
    cv2.rectangle(img, (rect[0] - MARGIN, rect[1] - MARGIN),
                  (rect[0] + rect[2] + MARGIN, rect[1] + rect[3] + MARGIN),
                  (0, 255, 0), 40)

# cv2.imwrite("../milestone1/30er3Mask.jpg", mask1)
final = stack_images(SIZE, [[img, mask1, mask2]])
cv2.imshow("finale", final)
cv2.waitKey(0)
