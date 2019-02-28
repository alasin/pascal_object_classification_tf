from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.contrib import eager as tfe

import util
import h5py

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class CaffeNet(keras.Model):
    def __init__(self, num_classes=10):
        super(CaffeNet, self).__init__(name='CaffeNet')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=96,
                                   kernel_size=[11, 11],
                                   strides=4,
                                   padding="valid",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)
        self.conv2 = layers.Conv2D(filters=256,
                                   kernel_size=[5, 5],
                                   strides=1,
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)
        self.conv3 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   padding="same",
                                   activation='relu')
        self.conv4 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   padding="same",
                                   activation='relu')
        self.conv5 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   padding="same",
                                   activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)



def vis_kernel(kernel):
    max_val = np.max(kernel)
    min_val = np.min(kernel)
    kernel = ((kernel - min_val)/(max_val - min_val))*256
    kernel = kernel.astype(np.uint8)
    plt.imshow(kernel)

def main():
    parser = argparse.ArgumentParser(description='Visualize CaffeNet')
    parser.add_argument('--ckpt', type=int, default=10,
                        help='input batch size for training')

    args = parser.parse_args()
    
    model = CaffeNet(num_classes=len(CLASS_NAMES))
    input_shape = tf.TensorShape([None, 224, 224, 3])
    model.build(input_shape)

    model.load_weights('pascal_caffenet/ckpt-' + str(args.ckpt))
    conv1_weights = model.get_layer(index=0).get_weights()[0]

    idxs_to_visualize = list(range(16))

    # conv1_weights = (conv1_weights / absMax) * 255

    
    for i, idx in enumerate(idxs_to_visualize):
        kernel = conv1_weights[:, :, :, idx]
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        vis_kernel(kernel)

    plt.show()


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
