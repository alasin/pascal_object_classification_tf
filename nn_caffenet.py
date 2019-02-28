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
from sklearn.manifold import TSNE
import matplotlib
from scipy.spatial.distance import cdist

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
        pool5_feat = self.pool3(x)
        flat_x = self.flat(pool5_feat)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        fc7_feat = self.dense2(out)
        out = self.dropout2(fc7_feat, training=training)
        out = self.dense3(out)
        return pool5_feat, fc7_feat

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)


def center_crop_test_data(x, y, z):
    x = tf.image.central_crop(x, central_fraction=0.875)
    return x, y, z


def test(model, dataset):
    all_pool5 = []
    all_fc7 = []
    all_images = []
    all_labels = []
    for batch, (images, labels, weights) in enumerate(dataset):
        pool5_feat, fc7_feat = model(images, training=False)
        all_pool5.append(pool5_feat.numpy())
        all_fc7.append(fc7_feat.numpy())
        all_labels.append(labels.numpy())
        all_images.append(images.numpy())
    
    all_pool5 = np.concatenate(all_pool5, axis=0)
    all_fc7 = np.concatenate(all_fc7, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    
    return all_pool5, all_fc7, all_images, all_labels


def main():
    parser = argparse.ArgumentParser(description='Visualize CaffeNet')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--ckpt', type=int, default=30,
                        help='input batch size for training')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()
    
    model = CaffeNet(num_classes=len(CLASS_NAMES))
    input_shape = tf.TensorShape([None, 224, 224, 3])
    model.build(input_shape)

    model.load_weights('pascal_caffenet/ckpt-' + str(args.ckpt))

    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')


    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(center_crop_test_data)
    test_dataset = test_dataset.batch(args.batch_size)

    pool5_feats, fc7_feats, all_images, all_labels = test(model, test_dataset)
    
    pool5_feats = np.reshape(pool5_feats, (pool5_feats.shape[0], -1))
    fc7_feats = np.reshape(fc7_feats, (fc7_feats.shape[0], -1))

    picked_classes = np.zeros((len(CLASS_NAMES)), dtype=np.uint8)
    
    i = 0
    num_images = 0
    images = []
    labels = []
    pool5_nns = []
    fc7_nns = []

    mean_rgb = np.array([123.68, 116.78, 103.94])
    all_images = np.add(all_images, mean_rgb)
    all_images = all_images.astype(np.uint8)
    
    while num_images < 10:
        label = np.argmax(all_labels[i], -1)
        if picked_classes[label]:
            i += 1
            continue
        
        images.append(all_images[i])
        labels.append(np.argmax(all_labels[i], -1))

        pool5 = np.expand_dims(pool5_feats[i], 0)
        fc7 = np.expand_dims(fc7_feats[i], 0)

        pool_dists = np.squeeze(cdist(pool5, pool5_feats))
        fc_dists = np.squeeze(cdist(fc7, fc7_feats))

        pool_idxs = list(np.argsort(pool_dists)[0:5])
        fc_idxs = list(np.argsort(fc_dists)[0:5])

        pool5_nns.append(pool_idxs)
        fc7_nns.append(fc_idxs)

        i += 1
        num_images += 1

    f, axarr = plt.subplots(num_images, 6)
    axarr[0, 0].set_title('Chosen image')
    for i in range(num_images):
        axarr[i, 0].imshow(images[i])
        axarr[i, 0].axis('off')

        for j in range(5):
            axarr[i, j+1].imshow(all_images[pool5_nns[i][j]])
            axarr[i, j+1].axis('off')
        
    f.suptitle('Pool5 NNs')
    plt.show()

    f, axarr = plt.subplots(num_images, 6)
    axarr[0, 0].set_title('Chosen image')
    for i in range(num_images):
        axarr[i, 0].imshow(images[i])
        axarr[i, 0].axis('off')

        for j in range(5):
            axarr[i, j+1].imshow(all_images[fc7_nns[i][j]])
            axarr[i, j+1].axis('off')
        
    f.suptitle('FC7 NNs')
    plt.show()

if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
