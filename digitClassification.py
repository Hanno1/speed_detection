import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


new_model = tf.keras.models.load_model('digitclassificationmodel.h5', custom_objects={'KerasLayer':hub.KerasLayer})
new_model.summary()
test_image = cv2.imread("normPictures/0number.png")
test_image = cv2.resize(test_image, (28, 28))
test_image = np.reshape(test_image, [1, 28, 28, 3])
classes = new_model.predict(test_image)
print(classes)
