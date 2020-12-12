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
    cv2.createTrackbar("Blue Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Green Min", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("Red Min", "Trackbars", 50, 255, empty)
    cv2.createTrackbar("Blue Max", "Trackbars", 160, 255, empty)
    cv2.createTrackbar("Green Max", "Trackbars", 110, 255, empty)
    cv2.createTrackbar("Red Max", "Trackbars", 255, 255, empty)

    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
        h_min = cv2.getTrackbarPos("Blue Min", "Trackbars")
        h_max = cv2.getTrackbarPos("Blue Max", "Trackbars")
        s_min = cv2.getTrackbarPos("Green Min", "Trackbars")
        s_max = cv2.getTrackbarPos("Green Max", "Trackbars")
        v_min = cv2.getTrackbarPos("Red Min", "Trackbars")
        v_max = cv2.getTrackbarPos("Red Max", "Trackbars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img, lower, upper)
        image_result = cv2.bitwise_and(img, img, mask=mask)

        final = stack_images(1, [[img, image_result]])
        cv2.imshow("Result", final)
        cv2.waitKey(1)


img = cv2.imread("normPictures/40norm.png")
trackbars(img)

