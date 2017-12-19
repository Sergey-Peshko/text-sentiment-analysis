import glob
import os

import numpy as np
from sklearn.utils import shuffle

import re

from DataSet import DataSet


def read_classes(train_path):
    lol = os.path.join(train_path, "*")
    files = glob.glob(lol)

    classes = []

    for fl in files:
        classes.append(os.path.basename(fl))

    return classes


def get_input_vector(text, vocab_path):
    with open(vocab_path) as f:
        vocab = f.read().splitlines()

    regex1 = re.compile(r"(<\\w+ \\/>)", re.IGNORECASE)
    regex2 = re.compile(r"[\\W\\d\\s&&[^']]", re.IGNORECASE)
    regex3 = re.compile(r" +", re.IGNORECASE)

    image = np.zeros(len(vocab))

    replaced = regex1.sub(" ", text)
    replaced = regex2.sub(" ", replaced)
    replaced = regex3.sub(" ", replaced)
    replaced = replaced.lower()

    words = replaced.split(" ")

    for word in words:
        try:
            index = vocab.index(word)
        except ValueError:
            index = -1
        if index is not -1:
            image[index] += 1

    return image


def load_class(class_path, vocab_size):
    images = []
    labels = []

    with open(class_path) as f:
        data = f.read().splitlines()

    for line in data:

        items = line.split(" ")
        labels.append(int(items.pop(0)))

        image = np.zeros(vocab_size, np.float16)

        for item in items:
            key_value = item.split(":")
            image[int(key_value[0])] = np.float16(key_value[1])

        images.append(image)

    return images, labels


def load_train(train_path, neg_file_name, pos_file_name, vocab_path):
    images = []
    labels = []

    with open(vocab_path) as f:
        vocab = f.read().splitlines()

    images_, labels_ = load_class(os.path.join(train_path, neg_file_name), len(vocab))
    images += images_
    labels += labels_

    images_, labels_ = load_class(os.path.join(train_path, pos_file_name), len(vocab))
    images += images_
    labels += labels_

    images = np.array(images, copy=False)
    labels = np.array(labels, copy=False)

    return images, labels, len(vocab)


def read_train_sets(train_path, neg_file_name, pos_file_name,  vocab_path, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, vocab_size = load_train(train_path, neg_file_name, pos_file_name, vocab_path)
    images, labels = shuffle(images, labels)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.valid = DataSet(validation_images, validation_labels)

    return data_sets, vocab_size
