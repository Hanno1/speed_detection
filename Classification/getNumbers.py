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

    contours, hierarchy = cv2.findContours(image_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    numbers = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if 2*w - VALUE < h < 2*w + VALUE and 10000 < w*h:
            print(w*h)
            number = image_gray[y:y+h, x:x+w]
            number = cv2.resize(number, (50, 100))
            numbers.append(number)
            # cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 0, 255), 3)

    for i in range(0, len(numbers)):
        cv2.imwrite("NumberExamples/" + str(i) + "number.png", numbers[i])
    """cv2.imshow("result", image_gray)
    cv2.waitKey(0)"""


def compare_numbers(number):
    """
    compare given number image to all numbers in compareNumbers
    """
    differences = []
    for i in range(0, 10):
        p = "compareNumbers/" + str(i) + ".png"
        image = cv2.imread(p)
        image_compare = cv2.bitwise_or(number, image)
        gray_original = np.sum(image) / 255
        gray_new = np.sum(image_compare) / 255

        difference = gray_new - gray_original
        differences.append(difference)
    print(differences)


VALUE = 100
path = "NumberExamples/1number.png"
img = cv2.imread(path)
compare_numbers(img)
# return_numbers(img)
