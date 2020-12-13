import cv2
import numpy as np


def edit_custom_image(image, gray=80):
    """
    sets size and set gray parameters for the given custom image """
    height = len(image[0])
    width = len(image[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for row in range(0, height):
        for col in range(0, width):
            if image[row][col] > gray:
                image[row][col] = 255
            else:
                image[row][col] = 0

    return image


def return_numbers(image):
    """
    returns a list of all numbers found in the image. (a list with the pictures of the numbers)
    to be classified in a neural network
    """
    image_gray = edit_custom_image(image)
    image_copy = image_gray.copy()

    image_blur = cv2.GaussianBlur(image_copy, (5, 5), 1)
    image_canny = cv2.Canny(image_blur, 100, 200)

    # invert gray image for neural network
    """for i in range(0, len(image_gray[0])):
        for j in range(0, len(image_gray[0])):
            image_gray[i][j] = 255 - image_gray[i][j]"""

    contours, hierarchy = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow("result", image_gray)
    cv2.waitKey(0)
    numbers = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w - VALUE < h < 2*w + VALUE and 5000 < w*h:
            cv2.rectangle(image_canny, (x, y), (x + w, y + h), (0, 0, 255), 3)
            number = image_gray[y:y+h, x:x+w]
            number = cv2.resize(number, (14, 26))
            numbers.append(number)
    overlay = np.zeros((28, 28, 3))
    for i in range(0, 28):
        for j in range(0, 28):
            overlay[i][j] = 255
    for i in range(0, len(numbers)):
        final_number = overlay
        final_number[1:27, 7:21] = numbers[i]
        cv2.imwrite("normPictures/" + str(i) + "number.png", final_number)


VALUE = 10
path = "normPictures/justTrying.png"
img = cv2.imread(path)
return_numbers(img)
