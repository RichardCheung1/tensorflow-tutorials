import tensorflow as tf
import numpy as np

# TF Learn  load the MNIST dataset from Yann Lecun's website
from tensorflow.examples.tutorials.mnist import input_data


# Convolutional Network helpers

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


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
Y = tf.placeholder(tf.float32, [batch_size, 10], name="class")

# weights and bias
# w initialized to random variables with mean of 0, stddev of 0.01
# b initialized to 0
W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01, name="weights"))
# W = tf.Variable(tf.zeros([784,10]))

'''
Changing weight does not have a significant impact on the accuracy of the model.
'''
b = tf.Variable(tf.zeros(shape=[1, 10], name="bias"))

# predict Y from X,w,b [linear]
logistic = tf.matmul(X, W) + b

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

'''
Using Convolutional  Network to improve accuracy
'''

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# shape of [5,5,1,32], where the first two are the patch size, the next is the number
# of input channel, and the last is the number of output channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape X into a 4D tensor
x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolve x_image with weight tensor, add the bias, and apply ReLU [rectifier linear unit]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# max_pool_2x2 reduces size by 2 -> 14x14
h_pool1 = max_pool_2x2(h_conv1)

# stack several layers
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout to reduce overfitting
# does not have much impact on smaller scale NN
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy_conv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy_conv)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy_conv = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
        # print("Accuracy {0}".format(total_correct_preds / MNIST.test.num_examples))

    '''
        The following lines uses the convolutional network approach
    '''
    for i in range(20000):
        batch = MNIST.test.next_batch(50)
        train_accuracy = accuracy_conv.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        if i % 100 == 0 :
            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #this statement prints 99.97%
    print("test accuracy %g" % accuracy_conv.eval(feed_dict={
        x: MNIST.test.images, y_: MNIST.test.labels, keep_prob: 1.0}))

