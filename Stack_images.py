import cv2
import numpy as np


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