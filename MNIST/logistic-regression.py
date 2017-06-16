import tensorflow as tf
import numpy as np

# TF Learn  load the MNIST dataset from Yann Lecun's website
from tensorflow.examples.tutorials.mnist import input_data

# MNIST contains   55,000 data points of training data
#                   10,000 data points of test data
#                   5,000 data points of validation data
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

# logistic regression batched

# determine parameters for model
learning_rate = 0.01
batch_size = 128
n_epochs = 25

# placeholder for features and labels
# each MNIST data is 28*28, which is 784
# there are 10 classes, corresponding to digits 0-9
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

# weights a nd bias
# w initialized to random variables with mean of 0, stddev of 0.01
# b initialized to 0
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01, name="weights"))
b = tf.Variable(tf.zeros(shape=[1, 10], name="bias"))

# predict Y from X,w,b [linear]
logistic = tf.matmul(X, w) + b

# loss function [cost]
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logistic, labels=Y, name="entropy")
# computes mean over examples in the  batch
loss = tf.reduce_mean(entropy)

# training op using gradient descent
# with given learning rate results an accuracy of 90.95%
# with learning rate of 0.001 results an accuracy of 87.07%
# with learning rate of 0.05 results and accuracy of 92.00%

# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.05).minimize(loss)

# trainning op using adam optimizer
# with default optimizer values results an accuracy of 92.73%
# with given learning rate above results an accuracy of 91.55%
optimizer = tf.train.AdamOptimizer().minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = int(MNIST.train.num_examples / batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})

        # test model
    n_batches = int(MNIST.test.num_examples / batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        _, loss_batch, logistic_batch = sess.run([optimizer, loss, logistic], feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logistic_batch)
        correct_preds = tf.equal(tf.arg_max(preds, 1), tf.arg_max(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

        print("Accuracy {0}".format(total_correct_preds / MNIST.test.num_examples))


