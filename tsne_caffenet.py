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
from sklearn.decomposition import PCA

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
        fc7_feat = self.dense2(out)
        out = self.dropout2(fc7_feat, training=training)
        out = self.dense3(out)
        return fc7_feat

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)


def center_crop_test_data(x, y, z):
    x = tf.image.central_crop(x, central_fraction=0.875)
    return x, y, z


def test(model, dataset):
    all_features = []
    all_labels = []
    for batch, (images, labels, weights) in enumerate(dataset):
        features = model(images, training=False)
        all_features.append(features.numpy())
        all_labels.append(labels.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels

def compute_average_labels(labels):
    num_images, num_dims = labels.shape
    output_labels = np.zeros(num_images)

    labels = labels.astype(np.uint8)
    
    for i in range(num_images):
        label_count = 0
        label_idx_count = 0.0
        for j in range(num_dims):
            if labels[i, j] == 1:
                label_count += 1
                label_idx_count += j
        
        if label_count > 0:
            output_labels[i] = label_idx_count/label_count
    
    return output_labels


def main():
    parser = argparse.ArgumentParser(description='Visualize CaffeNet')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()
    
    model = CaffeNet(num_classes=len(CLASS_NAMES))

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint('pascal_caffenet'))

    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    ordering = np.random.permutation(1000)
    test_images = test_images[ordering]
    test_labels = test_labels[ordering]
    test_weights = test_weights[ordering]


    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(center_crop_test_data)
    test_dataset = test_dataset.batch(args.batch_size)

    sample_feats, sample_labels = test(model, test_dataset)
    sample_label_colors = compute_average_labels(sample_labels)

    feats_embedded = TSNE(n_components=2).fit_transform(sample_feats)

    cmap = matplotlib.cm.get_cmap('tab20')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=19)
    plt.scatter(feats_embedded[:, 0], feats_embedded[:, 1], c=sample_label_colors, cmap=cmap)
   
    left, right = plt.xlim()
    plt.xlim(left, right + 5)
    recs = []
    for i in range(len(CLASS_NAMES)):
        recs.append(matplotlib.patches.Rectangle((0, 0), 1, 1, fc=cmap(norm(i))))

    plt.legend(recs, CLASS_NAMES, loc=4)
    plt.title('TSNE')
    plt.show()

    ######### PCA TSNE #######################
    pca = PCA(n_components=50)
    principal_components = pca.fit_transform(sample_feats)

    feats_embedded = TSNE(n_components=2).fit_transform(principal_components)

    cmap = matplotlib.cm.get_cmap('tab20')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=19)
    plt.scatter(feats_embedded[:, 0], feats_embedded[:, 1], c=sample_label_colors, cmap=cmap)
   
    left, right = plt.xlim()
    plt.xlim(left, right + 5)
    recs = []
    for i in range(len(CLASS_NAMES)):
        recs.append(matplotlib.patches.Rectangle((0, 0), 1, 1, fc=cmap(norm(i))))

    plt.legend(recs, CLASS_NAMES, loc=4)
    plt.title('TSNE with PCA (50-d)')
    plt.show()


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
