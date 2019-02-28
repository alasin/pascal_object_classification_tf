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

def mixup(images, labels, l_weights, alpha, batch_size):
    weights = np.random.beta(alpha, alpha, batch_size)
    
    image_weight = weights.reshape(batch_size, 1, 1, 1)
    label_weight = weights.reshape(batch_size, 1)
    l_w_weight = weights.reshape(batch_size, 1)
    ordering = np.random.permutation(batch_size)

    images_new = tf.gather(images, ordering)
    images_old = images

    labels_new = tf.gather(labels, ordering)
    labels_old = labels

    l_weights_new = tf.gather(l_weights, ordering)
    l_weights_old = l_weights

    images = image_weight * images_old + (1 - image_weight) * images_new
    labels = label_weight * labels_old + (1 - label_weight) * labels_new
    l_weights = l_w_weight * l_weights_old + (1 - l_w_weight) * l_weights_new

    return images, labels, l_weights



def augment_train_data(x, y, z):
    x = tf.image.random_crop(x, size=(224, 224, 3))
    x = tf.image.random_flip_left_right(x)
    return x, y, z

def center_crop_test_data(x, y, z):
    x = tf.image.central_crop(x, central_fraction=0.875)
    return x, y, z

def test(model, dataset):
    test_loss = tfe.metrics.Mean()
    test_accuracy = tfe.metrics.Accuracy()
    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images, training=False)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights)
        prediction = tf.round(tf.nn.sigmoid(logits))
        prediction = tf.cast(prediction, tf.int32)
        # prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, labels)
        test_loss(loss_value)
    return test_loss.result(), test_accuracy.result()



def main():
    parser = argparse.ArgumentParser(description='VGG Fine Tune Mixup')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=60,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval-interval', type=int, default=60,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log-dir', type=str, default='tb',
                        help='path for logging directory')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')


    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    train_dataset = train_dataset.map(augment_train_data)
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(center_crop_test_data)
    test_dataset = test_dataset.batch(args.batch_size)

    model = VGG(num_classes=len(CLASS_NAMES))

    

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    tf.contrib.summary.initialize()

    global_step = tf.train.get_or_create_global_step()


    train_log = {'iter': [], 'loss': [], 'accuracy': []}
    test_log = {'iter': [], 'loss': [], 'accuracy': []}

    ckpt_dir = 'pascal_vgg_mixup_weights'
    ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Build model first to load weights
    input_shape = tf.TensorShape([None, 224, 224, 3])
    model.build(input_shape)

    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    
    # Print layer names in saved weights 
    # f = h5py.File('vgg16_weights_tf_dim_ordering_tf_kernels.h5', 'r')

    # # Get the data
    # for i in list(f.keys()):
        # print(i)
    
    decayed_lr = tf.train.exponential_decay(args.lr, global_step, 1000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr(), momentum=0.9)

    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model)

    alpha = 0.2
    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        for batch, (images, labels, weights) in enumerate(train_dataset):
            batch_size = int(images.shape[0])
            labels = tf.cast(labels, tf.float32)
            weights = tf.cast(weights, tf.float32)

            images, labels, weights = mixup(images, labels, weights, alpha, batch_size)
            loss_value, grads = util.cal_grad(model,
                                              loss_func=tf.losses.sigmoid_cross_entropy,
                                              inputs=images,
                                              targets=labels,
                                              weights=weights)

            grads_and_vars = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars, global_step)
            
            epoch_loss_avg(loss_value)
            if global_step.numpy() % args.log_interval == 0:
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}'.format(ep,
                                                                                  args.epochs,
                                                                                  global_step.numpy(),
                                                                                  epoch_loss_avg.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())
                
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training Loss', loss_value)
                    tf.contrib.summary.image('RGB', images)
                    tf.contrib.summary.scalar('LR', decayed_lr())
                    
                    # for i, variable in enumerate(model.trainable_variables):
                    #     tf.contrib.summary.histogram("grad_" + variable.name, grads[i])
                
            if global_step.numpy() % args.eval_interval == 0:
                test_AP, test_mAP = util.eval_dataset_map(model, test_dataset)
                print("mAP: ", test_mAP)
                # print("Loss: %.4f, Acc: %.4f, mAP: %.4f", test_lotest_mAP)
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Test mAP', test_mAP)

        if ep % 2 == 0:
            root.save(ckpt_prefix)

    model.summary()
    

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
