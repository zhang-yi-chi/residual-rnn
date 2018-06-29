from __future__ import print_function
import tensorflow as tf
import os
import numpy as np


class ImageLoader(object):

    def __init__(self, path, batch_size=128, begin=0, end=10000):
        self.path = path
        self.batch_size = batch_size
        self._read_img_names(begin, end)

        self.batches_per_epoch = int(np.floor(len(self.img_paths) / batch_size))

        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices(self.img_paths)
        data = data.map(self._parse) #, num_parallel_calls=4)
        self.data = data.batch(batch_size)

    def _read_img_names(self, begin, end):
        _, __, imgs = next(os.walk(self.path))
        self.img_paths = list(map(lambda x: os.path.join(self.path, x), imgs[begin:end]))

    def _parse(self, filename):
        img_raw = tf.read_file(filename)
        img_decoded = tf.image.decode_image(img_raw, channels=3)
        img_croped = tf.random_crop(img_decoded, [32, 32, 3])
        img_flipped = tf.image.random_flip_left_right(img_croped)
        return img_flipped
