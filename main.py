# ----------------------------------
#
# This function shows how to use TensorFlow to
# create a soft margin SVM
#
# We will use the imdb data, specifically:
#  x1 = Sepal Length
#  x2 = Petal Width
# Class 1 : pos
# Class -1: neg
#
# We know here that x and y are linearly seperable
import datetime
import time

import tensorflow as tf
from tensorflow.python.framework import ops

from datasetutils import read_train_sets
from trainutils import train

number_epoch = 10000
learning_rate = 0.00001
alpha_val = 0.0001
batch_size = 1

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data
data_sets, num_inputs = read_train_sets("./aclImdb/train", "prepared_ned", "prepared_pos", "aclImdb/imdb.vocab", 0.1)

# Initialize placeholders
x_data = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32, name="x")
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y_true")

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[num_inputs, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare model operations
model_output = tf.subtract(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Declare loss function
# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# L2 regularization parameter, alpha
alpha = tf.constant([alpha_val])
# Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
# Put terms together
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# Declare prediction function
prediction = tf.sign(model_output, name="y_pred")
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Declare optimizer
my_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

report_file_name = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d_%H+%M+%S')

train(epoch_amount=number_epoch,
      data=data_sets,
      session=sess,
      x=x_data,
      y_true=y_target,
      batch_size=batch_size,
      accuracy=accuracy,
      cost=loss,
      optimizer=train_step,
      report_file_name=report_file_name)
