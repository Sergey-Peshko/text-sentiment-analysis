from DataSet import DataSet
from DataSetAccuracyManager import DataSetAccuracyManager
from datasetutils import load_class
import tensorflow as tf
import numpy as np
import os

# prepare_class("../aclImdb/test/neg/", "../aclImdb/imdb.vocab", "prepared_test_neg")
# prepare_class("../aclImdb/test/pos/", "../aclImdb/imdb.vocab", "prepared_test_pos")


with open("../aclImdb/imdb.vocab") as f:
    vocab = f.read().splitlines()

images, labels = load_class(os.path.join("../aclImdb/test/", "prepared_test_neg"), len(vocab))
images = np.array(images, copy=False)
labels = np.array(labels, copy=False)
neg_dataset = DataSet(images, labels)


images, labels = load_class(os.path.join("../aclImdb/test/", "prepared_test_pos"), len(vocab))
images = np.array(images, copy=False)
labels = np.array(labels, copy=False)
pos_dataset = DataSet(images, labels)


def test():
    # Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('../trained_model/model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('../trained_model'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")

    pos_data_accuracy_manager = DataSetAccuracyManager(pos_dataset, sess, x, y_true, 12500)
    neg_data_accuracy_manager = DataSetAccuracyManager(neg_dataset, sess, x, y_true, 12500)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

    print(pos_data_accuracy_manager.calc_accuracy(accuracy))
    print(neg_data_accuracy_manager.calc_accuracy(accuracy))


test()
