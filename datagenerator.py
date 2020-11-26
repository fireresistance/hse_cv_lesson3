import keras
import os
import random
import itertools
import numpy as np
import cv2
from imgaug import augmenters as iaa
from keras.utils import to_categorical
from keras.utils.data_utils import OrderedEnqueuer

class DataGenerator(keras.utils.Sequence):
    def __init__(self, src_dir, sample_per_class=100, uniq_classes="uniq_classes.txt", img_shape=(256,256), batch_size=4, shuffle=True, valid=False):
        self.images = []
        self.labels = []
        self.uniq_classes = open(uniq_classes).read().splitlines()
        #/home/redivan/datasets/dog_breeds/images
        #/home/redivan/datasets/dog_breeds/images/
        #C:\\tmp1\\
        #C:\\tmp1
        self.src_dir = src_dir
        self.src_dir = os.path.join(self.src_dir, '')
        self.amount_of_slashes = self.src_dir.count('/')
        self.sample_per_class = sample_per_class
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.valid = valid
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.images) / float(self.batch_size))

    def on_epoch_end(self):
        if not self.valid:
            self.images = []
            self.labels = []

            for folder, subs, files in os.walk(self.src_dir):
                checkfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
                if len(checkfiles) > 0:
                    counter = 0
                    if self.shuffle:
                        rnd = random.random() * 10000
                        random.Random(rnd).shuffle(checkfiles)
                    iter_files = itertools.cycle(checkfiles)
                    filename = next(iter_files)
                    tmpname = os.path.join(folder, filename)
                    entry = os.path.dirname(tmpname).split("/")[self.amount_of_slashes:]
                    breed = ''.join(x for x in entry)
                    if breed in self.uniq_classes:
                        while counter < self.sample_per_class:
                            filename = next(iter_files)
                            self.images.append(os.path.join(folder, filename))
                            self.labels.append(self.uniq_classes.index(breed))
                            counter +=1

            if self.shuffle:
                rnd = random.random() * 10000
                random.Random(rnd).shuffle(self.images)
                random.Random(rnd).shuffle(self.labels)

            self.images = np.array(self.images)
            self.labels = np.array(self.labels)
        else:
            self.images = []
            self.labels = []

            for folder, subs, files in os.walk(self.src_dir):
                checkfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
                if len(checkfiles) > 0:
                    counter = 0
                    if self.shuffle:
                        rnd = random.random() * 10000
                        random.Random(rnd).shuffle(checkfiles)
                    iter_files = itertools.cycle(checkfiles)
                    filename = next(iter_files)
                    tmpname = os.path.join(folder, filename)
                    entry = os.path.dirname(tmpname).split("/")[self.amount_of_slashes:]
                    breed = ''.join(x for x in entry)
                    if breed in self.uniq_classes:
                        while counter < 5:
                            filename = next(iter_files)
                            self.images.append(os.path.join(folder, filename))
                            self.labels.append(self.uniq_classes.index(breed))
                            counter += 1

            if self.shuffle:
                rnd = random.random() * 10000
                random.Random(rnd).shuffle(self.images)
                random.Random(rnd).shuffle(self.labels)

            self.images = np.array(self.images)
            self.labels = np.array(self.labels)

    def generate_data(self, indexs):

        images_batch = []
        labels_batch = []

        for i in indexs:
            image_name = os.path.join(self.src_dir, self.images[i])
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)

            if self.valid:
                seq = iaa.Sequential([])
            else:
                sometimes = lambda aug: iaa.Sometimes(0.75, aug)
                seq = iaa.Sequential(
                    [
                        iaa.Fliplr(0.5),
                        iaa.Crop(percent=(0, 0.1)),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.3))),
                        iaa.LinearContrast((0.75, 1.5)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05 * 255), per_channel=0.5),
                        iaa.Multiply((0.8, 1.2), per_channel=0.2)
                    ], random_order=True
                )
            seq_det = seq.to_deterministic()
            augmented_image = seq_det.augment_images([image])
            augmented_image = augmented_image[0]
            augmented_image = cv2.resize(augmented_image, self.img_shape)
            augmented_image = augmented_image / 255.
            images_batch.append(augmented_image)

            #image = cv2.resize(image, self.img_shape)
            #image = image / 255.

            image = image / 255.
            image = cv2.resize(image, self.img_shape)
            images_batch.append(image)

            labels_batch.append(to_categorical(self.labels[i], len(self.uniq_classes)))

        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        return images_batch, labels_batch

    def __getitem__(self, item):
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        a, la = self.generate_data(indexes)
        return a, la

if __name__ == "__main__":

    src_dir = "/home/redivan/datasets/dog_breeds/images"
    train_gen = DataGenerator(src_dir, img_shape=(512,512), uniq_classes="/home/redivan/datasets/dog_breeds/images/model_thr30.list", batch_size=16)
    enqueuer = OrderedEnqueuer(train_gen)
    enqueuer.start(workers=1, max_queue_size=4)
    output_gen = enqueuer.get()

    gen_len = len(train_gen)
    try:
        for i in range(gen_len):
            batch = next(output_gen)
            for a, la in zip(batch[0], batch[1]):
                print(a.shape)
                cv2.imshow("win", a)
                print(la)
                print(np.argmax(la))
                cv2.waitKey(0)
    finally:
        enqueuer.stop()