import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from model import build
from datagenerator import DataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    src_dir = "/home/redivan/datasets/dog_breeds/images"
    total_epochs = 100
    save_path = "./weights/"

    save_name = "{epoch:03d}-v_loss_{val_loss:.3f}.h5"
    checkpoint_classifier = ModelCheckpoint(os.path.join(save_path, save_name), monitor='acc', mode='max', save_best_only=False, verbose=0)

    feature_ext, model = build(num_classes=193)
    loss = "categorical_crossentropy"
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", "top_k_categorical_accuracy"])
    model.summary()
    train_gen = DataGenerator(src_dir, uniq_classes="/home/redivan/datasets/dog_breeds/images/model_thr30.list", batch_size=16)
    val_gen = DataGenerator(src_dir, uniq_classes="/home/redivan/datasets/dog_breeds/images/model_thr30.list", batch_size=16, shuffle=False, valid=True)
    model.fit_generator(train_gen, validation_data=val_gen, epochs=total_epochs, workers=8, use_multiprocessing=False, callbacks=[checkpoint_classifier])
