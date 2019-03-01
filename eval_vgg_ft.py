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

class VGG(keras.Model):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__(name='VGG')
        self.num_classes = num_classes
        self.block1_conv1 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   name='block1_conv1')
        self.block1_conv2 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   name='block1_conv2')
        self.block1_pool = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='block1_pool')
        
        self.block2_conv1 = layers.Conv2D(filters=128,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block2_conv1')
        self.block2_conv2 = layers.Conv2D(filters=128,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block2_conv2')
        self.block2_pool = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='block2_pool')
        
        self.block3_conv1 = layers.Conv2D(filters=256,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block3_conv1')
        self.block3_conv2 = layers.Conv2D(filters=256,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block3_conv2')
        self.block3_conv3 = layers.Conv2D(filters=256,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block3_conv3')
        self.block3_pool = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='block3_pool')
        
        self.block4_conv1 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block4_conv1')
        self.block4_conv2 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block4_conv2')
        self.block4_conv3 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block4_conv3')
        self.block4_pool = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='block4_pool')

        self.block5_conv1 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block5_conv1')
        self.block5_conv2 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block5_conv2')
        self.block5_conv3 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     name='block5_conv3')
        self.block5_pool = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='block5_pool')

        self.flat = layers.Flatten()

        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.final_fc = layers.Dense(num_classes)

    def call(self, inputs, training=True):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)

        flat_x = self.flat(x)
        out = self.fc1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.fc2(out)
        out = self.dropout2(out, training=training)
        out = self.final_fc(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)


def augment_train_data(x, y, z):
    x = tf.image.random_crop(x, size=(224, 224, 3))
    x = tf.image.random_flip_left_right(x)
    return x, y, z

def center_crop_test_data(x, y, z):
    x = tf.image.central_crop(x, central_fraction=0.875)
    return x, y, z

def test(model, dataset):
    test_loss = tfe.metrics.Mean()
    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images, training=False)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights)
        test_loss(loss_value)
    return test_loss.result()



def main():
    parser = argparse.ArgumentParser(description='Evaluate VGG FT')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()

    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')


    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(center_crop_test_data)
    test_dataset = test_dataset.batch(args.batch_size)

    model = VGG(num_classes=len(CLASS_NAMES))

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint('pascal_vgg_ft'))

    AP, mAP = util.eval_dataset_map(model, test_dataset)
    rand_AP = util.compute_ap(
        test_labels, np.random.random(test_labels.shape),
        test_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = util.compute_ap(test_labels, test_labels, test_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    print('Obtained {} mAP'.format(mAP))
    print('Per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, util.get_el(AP, cid)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
