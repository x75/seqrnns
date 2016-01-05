# this is from https://m.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/

import tensorflow as tf    
from tensorflow.models.rnn import rnn    
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    
import numpy as np
import matplotlib.pylab as pl

def gen_data(length=100):
    t = np.linspace(0, length, length, endpoint=False)
    s = np.zeros_like(t)
    for i in range(3):
        s += np.sin(t * (0.1 + np.random.uniform(-0.01, 0.01)))
    return s/np.max(np.abs(s))

def get_seq_input_data(ptr, data):
    # seq_input_data = np.random.rand(n_steps, batch_size, seq_width).astype('float32')
    seq_input_data = np.zeros((n_steps, batch_size, seq_width)).astype('float32')
    # seq_input_data[0, :, :] = 1.
    # seq_input_data[n_steps/2, :, :] = -1.
    for n in xrange(batch_size):
        # print n
        seq_input_data[:,n,:] = data[ptr:ptr+n_steps,np.newaxis]
        # pass
    
    return seq_input_data

if __name__ == '__main__':
    np.random.seed(1)      
    size = 100
    batch_size= 1 # 100
    n_steps = 500
    seq_width = 1

    data_pointer = 0
    data = gen_data(1000000)

    # initializer = tf.random_uniform_initializer(-0.8,0.8)
    initializer = tf.random_uniform_initializer(-0.00001,0.00001)
    # initializer = tf.zeros_initializer((size*2,1), dtype=tf.float32)

    seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    # sequence we will provide at runtime  
    early_stop = tf.placeholder(tf.int32)
    # what timestep we want to stop at

    inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
    # inputs for rnn needs to be a list, each item being a timestep. 
    # we need to split our input into each timestep, and reshape it because split keeps dims by default
    target = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    # targets = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, target)]
    # target = tf.placeholder(tf.float32, [None, None, seq_width])

    cell = LSTMCell(size, seq_width, initializer=initializer)  
    initial_state = cell.zero_state(batch_size, tf.float32)
    # initial_state = tf.random_uniform([batch_size, cell.state_size], -0.1, 0.1)
    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)
    # set up lstm
    final_state = states[-1]

    W_o = tf.Variable(tf.random_normal([size,1], stddev=0.01))
    b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

    print "type(outputs)", type(outputs)
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    output = tf.nn.xw_plus_b(output, W_o, b_o)
    output2 = tf.reshape(output, [n_steps, batch_size, seq_width])
    print "type(output)", type(output)
    # self.final_state = states[-1]

    # cost = output - target
    cost = tf.reduce_mean(tf.pow(output2 - target, 2)) # MSE
    learning_rate = 0.001
    # print "trainable", tf.trainable_variables()
    
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 10.)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.apply_gradients(zip(grads, tvars))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    # prev_state = session.run(cell.zero_state(batch_size, tf.float32))
    prev_state = session.run(tf.random_uniform([batch_size, cell.state_size], -1., 1.))

    allouts  = []
    allcosts = []
    # training
    for i in range(400):
        seq_input_data  = get_seq_input_data(data_pointer, data)
        seq_target_data = get_seq_input_data(data_pointer+1, data)

        feed = {early_stop: n_steps,
                seq_input: seq_input_data,
                initial_state: prev_state,
                target: seq_target_data}
            
        # fstate, _ = session.run([final_state, train_op], feed_dict=feed)
        session.run(optimizer, feed_dict=feed)
        # prev_state = fstate

        if i % 1 == 0:
            tcost = session.run(cost, feed_dict=feed)
            allcosts.append(tcost)
            # print len(tcost)
            print "cost[%d] = %f" % (i, tcost)
        
        data_pointer += n_steps

    pl.plot(allcosts)
    pl.show()
        
    # eval
    data_pointer = 0
    allouts  = []
    allcosts = []
    for i in range(3):
        seq_input_data  = get_seq_input_data(data_pointer, data)
        seq_target_data = get_seq_input_data(data_pointer+1, data)
        # print type(seq_target_data)
        data_pointer += n_steps
        # print "pstate", prev_state
        feed = {early_stop: n_steps,
                seq_input: seq_input_data,
                initial_state: prev_state,
                target: seq_target_data}
        # feed = {early_stop:n_steps, seq_input: seq_input_data}
        # define our feeds. 
        # early_stop can be varied, but seq_input needs to match the shape that was defined earlier

        outs, fstate, tcost = session.run([output, final_state, cost], feed_dict=feed)
        # outs, fstate, tcost, opt = session.run([output, final_state, cost, optimizer], feed_dict=feed)
        prev_state = fstate
        # print "fstate", fstate,tcost
        allouts.append(outs)
        allcosts.append(tcost)
        # tcost = session.run(cost, feed_d)
        if i % 10 == 0:
            print "cost[%d] = %f" % (i, tcost)
    
    # run once
    # output is a list, each item being a single timestep. Items at t>early_stop are all 0s
    # print outs
    print type(outs)
    print len(outs)
    print type(outs[0])
    print outs[0].shape
    print "allouts", len(allouts)

    pl.subplot(311)
    pl.plot(seq_input_data[:,0,:])
    pl.plot(seq_target_data[:,0,:])
    pl.plot(outs)
    pl.subplot(312)
    
    # for i,out in enumerate(allouts):
    #     print out.shape
    #     pl.plot(out)
    allouts = np.asarray(allouts)
    pl.plot(allouts.flatten())

    pl.subplot(313)
    pl.plot(allcosts)
    
    # pl.plot(outs)
    pl.show()
