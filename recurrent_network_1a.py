# this is from https://m.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/

import tensorflow as tf    
from tensorflow.models.rnn import rnn    
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    
import numpy as np
import matplotlib.pylab as pl

if __name__ == '__main__':
    np.random.seed(1)      
    size = 100
    batch_size= 1 # 100
    n_steps = 200
    seq_width = 1

    initializer = tf.random_uniform_initializer(-0.01,0.01) 
    # initializer = tf.zeros_initializer((size*2,1), dtype=tf.float32)

    seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    # sequence we will provide at runtime  
    early_stop = tf.placeholder(tf.int32)
    # what timestep we want to stop at

    inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
    # inputs for rnn needs to be a list, each item being a timestep. 
    # we need to split our input into each timestep, and reshape it because split keeps dims by default  

    cell = LSTMCell(size, seq_width, initializer=initializer)  
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)
    # set up lstm

    W_o = tf.Variable(tf.random_normal([size,1], stddev=0.01))
    b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

    print "type(outputs)", type(outputs)
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    output = tf.nn.xw_plus_b(output, W_o, b_o)
    # self.final_state = states[-1]

    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    # seq_input_data = np.random.rand(n_steps, batch_size, seq_width).astype('float32')
    seq_input_data = np.zeros((n_steps, batch_size, seq_width)).astype('float32')
    seq_input_data[0, :, :] = 1.
    seq_input_data[n_steps/2, :, :] = -1.
    feed = {early_stop:n_steps, seq_input: seq_input_data}
    # define our feeds. 
    # early_stop can be varied, but seq_input needs to match the shape that was defined earlier

    outs = session.run(output, feed_dict=feed)
    # run once
    # output is a list, each item being a single timestep. Items at t>early_stop are all 0s
    print outs
    print type(outs)
    print len(outs)
    print type(outs[0])
    print outs[0].shape

    pl.subplot(211)
    pl.plot(seq_input_data[:,0,:])
    pl.subplot(212)
    pl.plot(outs)
    pl.show()
