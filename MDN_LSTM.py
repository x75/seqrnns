
import matplotlib.pylab as pl
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

NSAMPLE = 10000
x_data = np.float32(np.linspace(0, 10*2*np.pi, NSAMPLE)).reshape((NSAMPLE, 1))
r_data = np.float32(np.random.normal(size=(NSAMPLE, 1)))
print x_data.shape
y_data = np.float32(0.5 * np.sin(2.77*x_data) + 0.9 * np.sin(3.13 * x_data) + r_data * 0.1)
print y_data.shape

pl.plot(x_data, y_data, "ro", alpha=0.3)
pl.show()

x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

NHIDDEN = 100
batch_size = 100
lstm_cell = rnn_cell.BasicLSTMCell(NHIDDEN, forget_bias=0.0)

initial_state = lstm_cell.zero_state(batch_size, tf.float32)

print lstm_cell

inputs = y

num_steps = 100
from tensorflow.models.rnn import rnn # state_saving_rnn
inputs = [tf.squeeze(input_, [1])
    for input_ in tf.split(1, num_steps, inputs)]
outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)

