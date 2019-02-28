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
        pool5_feat = self.block5_pool(x)

        flat_x = self.flat(pool5_feat)
        out = self.fc1(flat_x)
        out = self.dropout1(out, training=training)
        fc7_feat = self.fc2(out)
        out = self.dropout2(fc7_feat, training=training)
        out = self.final_fc(out)
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
    parser = argparse.ArgumentParser(description='Visualize VGG')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--ckpt', type=int, default=5,
                        help='input batch size for training')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()
    
    model = VGG(num_classes=len(CLASS_NAMES))
    input_shape = tf.TensorShape([None, 224, 224, 3])
    model.build(input_shape)

    model.load_weights('pascal_vgg_ft/ckpt-' + str(args.ckpt))

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
