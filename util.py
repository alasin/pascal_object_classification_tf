import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
import os
import skimage.io as sio
from skimage.transform import resize
import sys

def set_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    return session


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_pascal(data_dir, class_names, split='trainval'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        class_names (list): list of class names
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 256px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """

    npz_filename = data_dir + '/pascal_' + split + '.npz'
    if os.path.isfile(npz_filename):
        l = np.load(npz_filename)
        images_arr = l['images']
        labels_arr = l['labels']
        weights_arr = l['weights']
    else:
        num_classes = len(class_names)
        
        labels = {}
        weights = {}
        for i, class_name in enumerate(class_names):
            file_name = data_dir + '/' + 'ImageSets/Main/' + class_name + '_' + split + '.txt'
            f = open(file_name, 'r')
            for line in f:
                line = line[:-1]
                tokens = line.split(' ')
                idx = tokens[0]
                presence = tokens[-1]
                if idx not in labels:
                    labels[idx] = np.zeros(num_classes, dtype=np.int32)
                    weights[idx] = np.ones(num_classes, dtype=np.int32)
                
                if presence == '0' or presence == '1':
                    labels[idx][i] = 1

                if presence == '0':
                    weights[idx][i] = 1

            f.close()

        images_arr = []
        labels_arr = []
        weights_arr = []
        mean_rgb = np.array([123.68, 116.78, 103.94])
        for key, val in labels.items():
            im_name = data_dir + '/JPEGImages/' + key + '.jpg'
            im = sio.imread(im_name)
            im = resize(im, (256, 256, 3), preserve_range=True)
            im = np.subtract(im, mean_rgb)

            images_arr.append(im)
            labels_arr.append(val)
            weights_arr.append(weights[key])

        images_arr = np.array(images_arr, dtype=np.float32)
        labels_arr = np.array(labels_arr)
        weights_arr = np.array(weights_arr)

        np.savez(npz_filename, images=images_arr, labels=labels_arr, weights=weights_arr)
    
    return images_arr, labels_arr, weights_arr


def cal_grad(model, loss_func, inputs, targets, weights=1.0):
    """
    Return the loss value and gradients
    Args:
         model (keras.Model): model
         loss_func: loss function to use
         inputs: image inputs
         targets: labels
         weights: weights of the samples
    Returns:
         loss and gradients
    """

    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss_func(targets, logits, weights)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    ## TODO implement the code here
    gt = []
    preds = []
    valid = []
    for batch, (images, labels, weights) in enumerate(dataset):
        x = model(images, training=False)
        x = tf.nn.sigmoid(x)
        # x = tf.round(x)
        gt.append(labels.numpy())
        valid.append(weights.numpy())
        preds.append(x.numpy())
        # print(x[0], labels[0], weights[0])

    gt = np.concatenate(gt, axis=0)
    preds = np.concatenate(preds, axis=0)
    valid = np.concatenate(valid, axis=0)
    AP = compute_ap(gt, preds, valid)
    mAP = np.mean(AP)
    return AP, mAP


def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr

if __name__ == "__main__":
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    a, b, c = load_pascal('data/VOCdevkit/VOC2007', CLASS_NAMES, split='test')
