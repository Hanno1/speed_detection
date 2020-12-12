import numpy as np
import cv2


def get_bgr_values(image, lower, upper):
    """
    returns a image which has only color pixels in range of the lower, upper bgr-bounding """
    # creates an mask which filters through the image and sets values in the bounding box to 1 (else 0)
    mask = cv2.inRange(image, lower, upper)
    # now we get an colored image with the bitwise_and function
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def convert_original(image):
    """
   changes the original street signs to one which contains only the necessary parts (the numbers and not much of the
   red border) """
    image_copy = np.copy(image)
    image_copy = get_bgr_values(image_copy, np.array([0, 0, 190]), np.array([40, 40, 255]))

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.maxArea = 500000
    params.minArea = 10000

    params.filterByCircularity = True
    params.minCircularity = 0.9

    params.filterByConvexity = True
    params.minConvexity = 0.9

    params.filterByInertia = True
    params.minInertiaRatio = 0.9

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image_copy)

    image_cropped = image
    if keypoints:
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            r = int(keypoint.size) // 2
            image_cropped = image[x - r:x + r, y - r:y + r]

    cv2.imwrite(path, image_cropped)
    cv2.imshow("title", image_cropped)
    cv2.waitKey(0)


def compare(image, custom=True):
    """
    compare given image to 60 image
    """
    path0 = "normPictures/30norm.png"
    img0 = cv2.imread(path0)

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY)
    original_gray = np.sum(thresh1) / 255

    if not custom:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("result", thresh2)
        cv2.waitKey(0)
    else:
        thresh2 = image

    mask = cv2.bitwise_or(thresh1, thresh2)
    mask_gray = np.sum(mask) / 255

    diff = original_gray - mask_gray
    print(diff)

    cv2.imshow("compared", mask)
    cv2.waitKey(0)

    cv2.imwrite("../milestone2/Arbeit/classificationTry.png", mask)


def resizing():
    """
    resize every image in normPictures to size (500, 500)
    """
    for i in range(1, 14):
        p = "normPictures/" + str(i) + "0norm.png"
        image = cv2.imread(p)
        image = cv2.resize(image, (SIZE, SIZE))
        cv2.imwrite(p, image)
    p = "normPictures/5norm.png"
    image = cv2.imread(p)
    image = cv2.resize(image, (SIZE, SIZE))
    cv2.imwrite(p, image)


def edit_custom_image(image, gray=80):
    """
    sets size and set gray parameters for the given custom image """
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for row in range(0, SIZE):
        for col in range(0, SIZE):
            if image[row][col] > gray:
                image[row][col] = 255
            else:
                image[row][col] = 0

    """ cv2.imshow("winname", image)
    cv2.waitKey(0)"""

    return image


SIZE = 500

path = "normPictures/40norm.png"
img = cv2.imread(path)
img = edit_custom_image(img)

# resizing()
compare(img)
