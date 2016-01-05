'''

lets try and modify this to do simple regression for real valued output
the usual simple prediction/generation task for a timeseries



A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import sys
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import matplotlib.pylab as pl

# Import MINST data
# import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def next_batch(self, batch_size, idx, x, y):
    """Return the next `batch_size` examples from this data set."""
    # start = self._index_in_epoch
    # self._index_in_epoch += batch_size
    # if self._index_in_epoch > self._num_examples:
    #     # Finished epoch
    #     self._epochs_completed += 1
    #     # Shuffle the data
    #     perm = numpy.arange(self._num_examples)
    #     numpy.random.shuffle(perm)
    #     self._images = self._images[perm]
    #     self._labels = self._labels[perm]
    #     # Start next epoch
    #     start = 0
    #     self._index_in_epoch = batch_size
    #     assert batch_size <= self._num_examples
    # end = self._index_in_epoch
    # return self._images[start:end], self._labels[start:end]
    batch_x = x[idx:idx+batch_size]
    batch_y = y[idx:idx+batch_size]


# print type(mnist)
# get my own data: MSO
ndim = 1
tau = 2 * np.pi
NUMSAMPLES = 10000
t = np.linspace(0, tau, NUMSAMPLES)
sin = np.sum(c * np.sin(tau * f * t) for c, f in ((2, 1.5), (3, 1.8), (4, 1.1))).reshape((NUMSAMPLES, ndim))

# print("sin.shape", sin.shape)

x = sin[:-1]
y = sin[1:] # shifted input

# pl.plot(sin)
# pl.plot(x)
# pl.plot(y)
# pl.show()

# sys.exit()

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = NUMSAMPLES / 100
display_step = 10

# Network Parameters
n_input = 1 # MNIST data input (img shape: 28*28)
n_steps = 300 # timesteps
n_hidden = 30 # hidden layer num of features
# n_classes = 10 # MNIST total classes (0-9 digits)
n_output = n_input # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_output])

# # Define weights
# weights = {
#     'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
#     'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
# }
# biases = {
#     'hidden': tf.Variable(tf.random_normal([n_hidden])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

with tf.variable_scope('rnnlm'):
    output_w = tf.get_variable("output_w", [n_hidden, n_output])
    output_b = tf.get_variable("output_b", [n_output])
    
def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    # _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    # return tf.matmul(outputs[-1], _weights['out']) + _biases['out']
    return tf.matmul(outputs[-1], output_w) + output_b

pred = RNN(x, istate, output_w, output_b)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
cost = tf.reduce_mean(tf.nn.l2_loss((y - pred))) # MSE loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))

# Initializing the variables
init = tf.initialize_all_variables()

print("blub")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    didx = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_x = x[didx:didx+batch_size]
        batch_y = y[didx:didx+batch_size]
        didx += batch_size
        # Reshape data to get 28 seq of 28 elements
        # batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
        if step % display_step == 0:
            # # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
            #                                     istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss)# + \
            # ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # # Calculate accuracy for 256 mnist test images
    # test_len = 256
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                                          istate: np.zeros((test_len, 2*n_hidden))})
