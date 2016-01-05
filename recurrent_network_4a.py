'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Modified: Oswald Berthold
Rewritten for single step 1-dimensional prediction task

'''

# Import MINST data
import matplotlib.pylab as pl
# import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# print type(mnist)

class SineWave(object):
    def __init__(self, length, freq = 1.):
        self.length = length
        self.t = np.linspace(0, 2*np.pi, self.length)
        self.s = np.sin(self.t * freq)

    def next_batch(self, batchsize=1):
        index = np.random.randint(self.length - batchsize - 1)
        x = self.s[index:index+batchsize].reshape((batchsize, 1))
        y = self.s[index+1:index+batchsize+1].reshape((batchsize, 1))
        return x, y

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np


sinewave = SineWave(100000, freq = 100.)


# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 1 # 28 # MNIST data input (img shape: 28*28)
n_steps = 1 # 28 # timesteps
n_hidden = 128 # hidden layer num of features
# n_classes = 10 # MNIST total classes (0-9 digits)
n_output = 1

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_output])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out'], states[-1]

pred, state = RNN(x, istate, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
cost = tf.reduce_mean(tf.pow(pred - y, 2)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
# optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay=0.001).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))

# Initializing the variables
init = tf.initialize_all_variables()

losses = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # print("batch_xs.shape", batch_xs.shape)
        # print("batch_ys.shape", batch_ys.shape)
        # # Reshape data to get 28 seq of 28 elements
        # batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # print("batch_xs.shape", batch_xs.shape)

        # my input data
        batch_xs, batch_ys = sinewave.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        
        # Fit training using batch data
        _, lstate = sess.run([optimizer,state], feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})
            losses.append(loss)
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    
    test_x = sinewave.s[1000:5000].reshape((-1, n_steps, n_input))
    test_y = sinewave.s[1001:5001].reshape((-1, n_input))
    epred, estate = sess.run([pred, state], feed_dict = {x: test_x, y: test_y, istate: np.zeros((4000, 2 * n_hidden))})
    # fpred, fstate = sess.run([pred, state], feed_dict = {x: test_x, y: test_y, istate: lstate})
    print pred, epred # , fpred

    pl.subplot(211)
    pl.plot(test_y, label="y")
    pl.plot(epred, label="epred")
    # pl.plot(fpred, label="fpred")
    pl.legend()
    pl.subplot(212)
    pl.plot(losses, label="loss")
    pl.legend()
    pl.show()
    
    # # Calculate accuracy for 256 mnist test images
    # test_len = 256
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                                          istate: np.zeros((test_len, 2*n_hidden))})
