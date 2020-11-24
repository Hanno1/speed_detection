import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_gray_scale(image):
    """
   prints the gray scale distribution of the image as a histogram """
    gray_scale = [0 for i in range(0, 256)]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # for every entry
    print(np.shape(image_gray))
    row_length = np.shape(image_gray)[0]
    col_length = np.shape(image_gray)[1]
    print(row_length, col_length)
    for row in range(0, row_length):
        for col in range(0, col_length):
            # get the gray value at the point (row, col)
            gray_val = image_gray[row][col]
            # add 1 to the existing gray values in gray_scale vector
            gray_scale[gray_val] += 1

    print(gray_scale)
    # show vector
    x_axis = [x for x in range(0, 256)]
    y_axis = gray_scale

    plt.plot(x_axis, y_axis, 'ro')
    plt.axis([0, 256, 0, 10000])
    plt.show()


img = cv2.imread("../milestone1/pictures/80er.jpg")
print(np.shape(img))
get_gray_scale(img)
