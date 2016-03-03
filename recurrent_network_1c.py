# this is from https://m.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/

# test cwrnn (without training) to verify temporal segmentation

import tensorflow as tf    
from tensorflow.models.rnn import rnn    
from tensorflow.models.rnn.rnn_cell import BasicRNNCell, BasicLSTMCell, LSTMCell, CWRNNCell
import numpy as np
import matplotlib.pylab as pl

def get_seq_input_data():
    # seq_input_data = np.random.rand(n_steps, batch_size, seq_width).astype('float32')
    seq_input_data = np.zeros((n_steps, batch_size, seq_width)).astype('float32')
    seq_input_data[0, :, :] = 1.
    seq_input_data[n_steps/2, :, :] = -1.
    return seq_input_data

if __name__ == '__main__':
    np.random.seed(1)      
    size = 16
    batch_size= 1 # 100
    n_steps = 200
    seq_width = 1

    initializer = tf.random_uniform_initializer(-0.8,0.8)
    # initializer = tf.zeros_initializer((size*2,1), dtype=tf.float32)

    seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    # sequence we will provide at runtime  
    early_stop = tf.placeholder(tf.int32)
    # what timestep we want to stop at

    inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
    # inputs for rnn needs to be a list, each item being a timestep. 
    # we need to split our input into each timestep, and reshape it because split keeps dims by default
    # result = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    result = tf.placeholder(tf.float32, [None, seq_width])
    

    # cell = LSTMCell(size, seq_width, initializer=initializer)  
    # cell = CWRNNCell(size, [1, 4, 16, 64])#, seq_width, initializer=initializer)
    cell = CWRNNCell(size, [1, 2, 4, 8])#, seq_width, initializer=initializer)
    # cell = BasicRNNCell(size)#, seq_width, initializer=initializer)
    # initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.random_uniform([batch_size, cell.state_size], -0.1, 0.1)
    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state) #, sequence_length=early_stop)
    # set up lstm
    final_state = states[-1]

    W_o = tf.Variable(tf.random_normal([size,1], stddev=0.01))
    b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

    print "type(outputs)", type(outputs)
    output_cat = tf.reshape(tf.concat(1, outputs), [-1, size])
    output = tf.nn.xw_plus_b(output_cat, W_o, b_o)
    # self.final_state = states[-1]
    output2 = tf.reshape(output, [batch_size, n_steps, seq_width])
    output2 = output2 + tf.random_normal([batch_size, n_steps, seq_width], stddev=0.05)
    # then transpose
    output2 = tf.transpose(output2, [1, 0, 2])
    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    seq_input_data = get_seq_input_data()
    
    # prev_state = session.run(cell.zero_state(batch_size, tf.float32))
    prev_state = session.run(tf.random_uniform([batch_size, cell.state_size], -1., 1.))

    allouts = []
    allstates = []
    allhiddens = []
    for i in range(1):
        print "pstate", prev_state
        feed = {early_stop:n_steps, seq_input: seq_input_data, initial_state: prev_state}
        # feed = {early_stop:n_steps, seq_input: seq_input_data}
        # define our feeds. 
        # early_stop can be varied, but seq_input needs to match the shape that was defined earlier

        outs, fstate, hidden = session.run([output, final_state, output_cat], feed_dict=feed)
        print type(outs), type(fstate), type(hidden)
        prev_state = fstate
        print "fstate", fstate
        allouts.append(outs)
        allstates.append(fstate)
        allhiddens.append(hidden)
    
    # run once
    # output is a list, each item being a single timestep. Items at t>early_stop are all 0s
    # print outs
    print type(outs)
    print len(outs)
    print type(outs[0])
    print outs[0].shape
    print "allouts", len(allouts)

    pl.subplot(411)
    pl.plot(seq_input_data[:,0,:])
    for i,out in enumerate(allouts):
        print out.shape
        pl.subplot(412)
        pl.plot(out)
        pl.subplot(413)
        pl.plot(allstates[i])
        pl.subplot(414)
        print "hidden[i].shape", hidden[i].shape
        cols = ["k", "b", "r", "g"]
        for j in range(0, len(hidden[i]), size / 4):
        # for j in range(0, 8, 4):
            print j, j/4, allhiddens[i]
            pl.plot(allhiddens[i][:,j:j+4], cols[j/4] + "-")
    # pl.plot(outs)
    pl.show()
