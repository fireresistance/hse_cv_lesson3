import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from model import build
import cv2
import numpy as np

if __name__ == "__main__":

    src_name = "1.jpg"
    weights_path = "weights/004-v_loss_2.321.h5"
    class_names = open("class.names").read().splitlines()

    img_size = (256, 256, 3)
    feature_ext, model = build(193, img_size)
    model.load_weights(weights_path)

    img = cv2.imread(src_name)
    img = cv2.resize(img, (256, 256))
    img = img / 255.

    load_img = np.expand_dims(img, axis=0)
    preds = model.predict(load_img)
    index = np.argmax(preds[0])
    top_five_indexes = preds[0].argsort()[-5:][::-1]
    top_class = class_names[index]
    top_prob = preds[0][index]
    for i in top_five_indexes:
        print("class: {}, prob: {}".format(class_names[i], float(preds[0][i])))

    print(1)


