import os

import tensorflow as tf

from datasetutils import get_input_vector


text = "good film"


def load_graph(trained_model):
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )
    return graph


def predict(text):
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = get_input_vector(text, "aclImdb/imdb.vocab")

    x_batch = x_batch.reshape((1, len(x_batch)))

    graph = load_graph(os.path.dirname(__file__) + '/trained_model/model.pb')

    y_pred = graph.get_tensor_by_name("y_pred:0")
    # Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")

    sess = tf.Session(graph=graph)

    # Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    print(result)


predict(text)
