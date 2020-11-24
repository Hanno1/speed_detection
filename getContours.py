import cv2
import numpy as np


MARGIN = 0
VALUE = 50
WIDTH = 500
HEIGHT = 500


def empty(x):
    """
    only because the trackbars need an function """
    pass


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


def trackbars(image):
    """
    creates an trackbar window and an image and
    shows points in the image according to the position of the trackbars
    """
    # creates trackbars for an image in hsv format
    # 3 channels: hue, saturation, value
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 179, 255, empty)

    # transform the image into the hsv
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # we constantly want to refresh the image (while the trackbars are moving)
    while True:
        # get the position of all 6 trackbars
        h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
        h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
        s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
        s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
        v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
        v_max = cv2.getTrackbarPos("Val Max", "Trackbars")
        # create an numpy array with the lower and upper position of the trackbars
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        # creates an mask which has value 1 if the image point is in the bounding box defined by lower, upper
        mask = cv2.inRange(img_hsv, lower, upper)
        # takes the original image and use the mask on it
        image_result = cv2.bitwise_and(image, image, mask=mask)

        final = stack_images(0.5, [[image, img_hsv], [mask, image_result]])
        cv2.imshow("Result", final)
        cv2.waitKey(1)


def show_contours(image):
    """
    takes a image and (hopefully) get the contours of the shield """
    # lower and upper bounding in bgr format for the color of the shield
    lower = np.array([0, 0, 50])
    upper = np.array([120, 80, 255])
    # creates an mask which filters through the image and sets values in the bounding box to 1 (else 0)
    mask = cv2.inRange(image, lower, upper)

    # now we get an colored image with the bitwise_and function
    image_result = cv2.bitwise_and(image, image, mask=mask)

    # just for research probably (:
    image_hsv = cv2.cvtColor(image_result, cv2.COLOR_BGR2HSV)
    # image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)
    # trackbars(image_result)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 110])
    mask = cv2.inRange(image_hsv, lower, upper)
    mask = cv2.bitwise_not(mask)
    # creates an rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilates the mask with the kernel
    mask = cv2.dilate(mask, kernel=kernel, iterations=1)
    # smooth the image with a gaussian filter
    image_blur = cv2.GaussianBlur(mask, (5, 5), 1)

    # save a contour which could be a shield in the following points
    # w means width and h is height
    x_biggest = y_biggest = w_biggest = h_biggest = 0
    # get the contours
    contours, hierachy = cv2.findContours(image_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # since the area is not meaningful we want the biggest width
    max_peri = 0
    # for every contour
    for cnt in contours:
        # get area, perimeter, approximation and vertices
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        vertices = len(approx)
        # get a bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        # if the contour is almost a square, the area is bigger then 100 and it has more then 4 vertices
        if w - VALUE < h < w + VALUE and area > 100 and vertices > 4:
            # we want the maximum perimeter because area is not that reliable
            if peri > max_peri:
                # save the coordinates
                x_biggest = x
                y_biggest = y
                w_biggest = w
                h_biggest = h
                # revalue the maximum perimeter
                max_peri = peri
            # draws an rectangle that satisfies the entire if conditions
            cv2.rectangle(image, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), (0, 255, 0), 2)

    # draws an rectangle around the biggest perimeter with a margin MARGIN
    cv2.rectangle(image, (x_biggest - MARGIN, y_biggest - MARGIN),
                         (x_biggest + w_biggest + MARGIN, y_biggest + h_biggest + MARGIN),
                         (0, 0, 255), 10)
    # takes all four corner points and brings them in order to warp
    pts1 = np.float32([[x_biggest - MARGIN, y_biggest - MARGIN], [x_biggest + w_biggest + MARGIN, y_biggest - MARGIN],
                       [x_biggest - MARGIN, y_biggest + h_biggest + MARGIN], [x_biggest + w_biggest + MARGIN, y_biggest + h_biggest + MARGIN]])
    pts2 = np.float32([[0, 0], [w_biggest + 2 * MARGIN, 0],
                       [0, h_biggest + 2 * MARGIN], [w_biggest+ 2 * MARGIN, h_biggest + 2 * MARGIN]])

    warp_perspective = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, warp_perspective, (w_biggest + 2 * MARGIN, h_biggest + 2 * MARGIN))
    dst = cv2.resize(dst, (WIDTH, HEIGHT))

    result = stack_images(0.1, [[image, image_result, image_hsv], [mask, image_blur, image]])
    cv2.imshow("Result", dst)
    cv2.imshow("all Pictures", result)
    cv2.waitKey(0)


path = "pictures/20201122_124059.jpg"
# reads and saves the image in img
img = cv2.imread(path)
show_contours(img)
