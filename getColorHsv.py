import numpy as np
import cv2


def empty(x):
    return x


def stack_images(scale, image_array):
    image_scale = 1 / scale
    for r in range(0, len(image_array)):
        for i in range(0, len(image_array[0])):
            image_array[r][i] = cv2.resize(image_array[r][i], (int(image_array[r][i].shape[1] // image_scale),
                                                               int(image_array[r][i].shape[0] // image_scale)))
            if len(image_array[r][i].shape) == 2:
                image_array[r][i] = cv2.cvtColor(image_array[r][i], cv2.COLOR_GRAY2BGR)
            elif len(image_array[r][i].shape) != 3:
                raise ValueError
            else:
                pass

    first_image_row = image_array[0][0]
    for i in range(1, len(image_array[0])):
        first_image_row = np.hstack((first_image_row, image_array[0][i]))

    for row in range(1, len(image_array)):
        image_row = image_array[row][0]
        for i in range(1, len(image_array[row])):
            image_row = np.hstack((image_row, image_array[row][i]))
        first_image_row = np.vstack((first_image_row, image_row))

    return first_image_row


def trackbars(img):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    cv2.createTrackbar("Hue Min", "Trackbars", 160, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
        h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
        h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
        s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
        s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
        v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
        v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        image_result = cv2.bitwise_and(img, img, mask=mask)

        final = stack_images(0.15, [[img, img_hsv], [mask, image_result]])
        cv2.imshow("Result", final)
        cv2.waitKey(1)


img = cv2.imread("pictures/30er5.jpg")
trackbars(img)
