# this is from https://m.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/

# create a tf RNN of type X

# train network
# python recurrent_network_1b.py -m train --seq_length 100 --train_step 1000

# sample from network with fixed input
# python recurrent_network_1b.py -m sample --seq_length 100 --train_step 1000

# sample from network in freerunning mode
# python recurrent_network_1b.py -m sample_fr --seq_length 100 --train_step 1000

import argparse
import tensorflow as tf    
from tensorflow.models.rnn import rnn    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn.rnn_cell import BasicRNNCell, BasicLSTMCell, LSTMCell, CWRNNCell
import numpy as np
import matplotlib.pylab as pl
import cPickle

def gen_data(length=100):
    t = np.linspace(0, length, length, endpoint=False)
    s = np.zeros_like(t)
    for i in range(3):
        s += np.sin(t * (((i+1) * 0.1) + np.random.uniform(-0.01, 0.01)))
        # s += np.sin(t * (((i+1) * 0.1)))
    return s/np.max(np.abs(s))

def get_seq_input_data(ptr, data, p):
    # seq_input_data = np.random.rand(n_steps, batch_size, seq_width).astype('float32')
    seq_input_data = np.zeros((p["n_steps"], p["batch_size"], p["seq_width"])).astype('float32')
    # seq_input_data[0, :, :] = 1.
    # seq_input_data[n_steps/2, :, :] = -1.
    for n in xrange(p["batch_size"]): # minus one?
        # print "batch#", n, seq_input_data[:,n,:].shape, data[ptr:ptr+p["n_steps"],np.newaxis].shape
        seq_input_data[:,n,:] = data[ptr:ptr+p["n_steps"],np.newaxis]
        # pass
    
    return seq_input_data

class Model():
    def __init__(self, n_steps = 100, batch_size = 1):
        np.random.seed(5)
        self.size = 300
        self.batch_size = batch_size # 100
        self.n_steps = n_steps
        self.seq_width = 1

        # initializer = tf.random_uniform_initializer(-0.8,0.8)
        initializer = tf.random_uniform_initializer(-0.00001,0.00001)
        # initializer = tf.zeros_initializer((size*2,1), dtype=tf.float32)

        self.seq_input = tf.placeholder(tf.float32, [self.n_steps, self.batch_size, self.seq_width])
        # seq_input = tf.placeholder(tf.float32, [n_steps, None, seq_width])
        # sequence we will provide at runtime  
        self.early_stop = tf.placeholder(tf.int32)
        # what timestep we want to stop at

        # inputs for rnn needs to be a list, each item being a timestep. 
        # we need to split our input into each timestep, and reshape it because split keeps dims by default
        self.inputs = [tf.reshape(i, (self.batch_size, self.seq_width)) for i in tf.split(0, self.n_steps, self.seq_input)]
        
        # inputs = tf.split(0, n_steps, seq_input)
        self.target = tf.placeholder(tf.float32, [self.n_steps, self.batch_size, self.seq_width])
        # target = tf.placeholder(tf.float32, [n_steps, None, seq_width])
        # targets = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, target)]
        # target = tf.placeholder(tf.float32, [None, None, seq_width])

        # cell = BasicRNNCell(self.size)
        # cell = LSTMCell(self.size, self.seq_width, initializer=initializer) # problem with multirnncell
        # cell = BasicLSTMCell(self.size, forget_bias=5.0)
        cell = CWRNNCell(self.size, [1, 4, 16, 64, 128])#, seq_width, initializer=initializer)

        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        if self.n_steps > 1: # training mode
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.8)

        self.cell = cell
    
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        # initial_state = tf.random_uniform([batch_size, cell.state_size], -0.1, 0.1)
        # outputs, states = rnn.rnn(self.cell, self.inputs, initial_state = self.initial_state, sequence_length = self.early_stop)
        outputs, states = rnn.rnn(self.cell, self.inputs, initial_state = self.initial_state)
        # set up lstm
        self.final_state = states[-1]

        W_o = tf.Variable(tf.random_normal([self.size,1], stddev=0.01))
        b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

        # now, outputs is a list with len = seqlen and elems of dim batchsize x seqwidth
        output = tf.reshape(tf.concat(1, outputs), [-1, self.size])
        output = tf.tanh(tf.nn.xw_plus_b(output, W_o, b_o))
        # get it right here
        self.output2 = tf.reshape(output, [self.batch_size, self.n_steps, self.seq_width])
        # apply output noise?
        # self.output2 = self.output2 + tf.random_normal([self.batch_size, self.n_steps, self.seq_width], stddev=0.001)
        
        # then transpose
        self.output2 = tf.transpose(self.output2, [1, 0, 2])
        # self.final_state = states[-1]

        # cost = output - target
        # self.cost = tf.reduce_mean(tf.pow(self.output2 - self.target, 2)) # MSE
        self.cost = tf.reduce_mean(tf.abs(self.output2 - self.target)) # MAE
        
        learning_rate = 0.001 # 0.001
        # print "trainable", tf.trainable_variables()
    
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 10.)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay=0.9, momentum=0.9, epsilon=1e-10).minimize(self.cost) # Adam Optimizer
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost) # Adam Optimizer

def train(args):
    data_pointer = 0
    data = gen_data(args.seq_length * args.train_steps + 1)

    pl.plot(data)
    pl.show()

    model = Model(n_steps = args.seq_length, batch_size = args.batch_size)
    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    # with tf.device("/gpu:1"):
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    saver = tf.train.Saver(tf.all_variables(), max_to_keep = 100)
    
    # prev_state = session.run(cell.zero_state(batch_size, tf.float32))
    prev_state = session.run(tf.random_uniform([model.batch_size, model.cell.state_size], -1., 1.))

    allouts  = []
    allcosts = []
    params = {
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "seq_width": model.seq_width
        }
    # training
    # pl.ion()
    # pl.figure()
    for i in range(args.train_steps):
        seq_input_data  = get_seq_input_data(data_pointer, data, params)
        seq_target_data = get_seq_input_data(data_pointer+1, data, params)

        feed = {model.early_stop: model.n_steps,
                model.seq_input: seq_input_data,
                model.initial_state: prev_state,
                model.target: seq_target_data}
            
        # fstate, _ = session.run([final_state, train_op], feed_dict=feed)
        session.run(model.optimizer, feed_dict=feed)
        # oS, o, o3 = session.run([outputs, output, output2], feed_dict=feed)
        # oS = session.run(outputs, feed_dict=feed)
        # o1 = session.run(output, feed_dict=feed)
        # o2 = session.run(output2, feed_dict=feed)
        # # tg = session.run(target, feed_dict=feed)
        # tg = seq_target_data.copy()
        # print type(oS), type(o1), type(o2)
        # print "o1.shape", o1.shape
        # print "o2.shape", o2.shape
        # print "tg.shape", tg.shape

        # pl.subplot(311)
        # pl.cla()
        # pl.plot(o1)
        # pl.subplot(312)
        # pl.cla()
        # for j in range(batch_size):
        #     pl.plot(o2[:,j,:])
        # pl.subplot(313)
        # pl.cla()
        # for j in range(batch_size):
        #     pl.plot(tg[:,j,:])
        # pl.draw()
        # prev_state = fstate

        if i % 1 == 0:
            tcost = session.run(model.cost, feed_dict=feed)
            allcosts.append(tcost)
            # print len(tcost)
            print "cost[%d] = %f" % (i, tcost)
        if i % 100 == 0:
            saver.save(session, "recurrent_network_1b.ckpt", global_step = i)
        
        data_pointer += model.n_steps

    pl.ioff()
    pl.title("training cost")
    pl.plot(allcosts)
    pl.show()
        
    # saver.save(session, "recurrent_network_1b.ckpt", global_step = i)
    f = open("recurrent_network_1b_allcosts.cpkl", "wb")
    cPickle.dump(allcosts, f)
    f.close()
    saver.save(session, "recurrent_network_1b.ckpt")

def sample(args):
    data_pointer = 0
    data = gen_data(args.seq_length * args.train_steps + 1)

    # pl.plot(data)
    # pl.show()
    
    model = Model(n_steps = args.seq_length, batch_size = args.batch_size)
    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    # with tf.device("/gpu:1"):
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    saver = tf.train.Saver(tf.all_variables(), max_to_keep = 100)

    saver.restore(session, "recurrent_network_1b.ckpt")
    
    # eval
    prev_state = session.run(tf.random_uniform([model.batch_size, model.cell.state_size], -1., 1.))
    data_pointer = 0
    allouts  = []
    allcosts = []
    params = {
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "seq_width": model.seq_width
        }
    for i in range(args.train_steps):
        seq_input_data  = get_seq_input_data(data_pointer, data, params)
        seq_target_data = get_seq_input_data(data_pointer+1, data, params)

        # pl.plot(seq_input_data[:,0,:])
        # pl.plot(seq_target_data[:,0,:])
        # pl.show()
        
        # print type(seq_target_data)
        data_pointer += model.n_steps
        # print "pstate", prev_state
        feed = {model.early_stop: model.n_steps,
                model.seq_input: seq_input_data,
                model.initial_state: prev_state,
                model.target: seq_target_data}
        # feed = {early_stop:n_steps, seq_input: seq_input_data}
        # define our feeds. 
        # early_stop can be varied, but seq_input needs to match the shape that was defined earlier

        outs, fstate, tcost = session.run([model.output2, model.final_state, model.cost], feed_dict=feed)
        # outs, fstate, tcost, opt = session.run([output, final_state, cost, optimizer], feed_dict=feed)
        prev_state = fstate
        # print "fstate", outs,fstate,tcost
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
    print outs.shape, outs[0].shape
    print "allouts", len(allouts)

    pl.subplot(311)
    pl.title("input, target, output")
    for j in range(model.batch_size):
        print "batch#", j
        pl.plot(seq_input_data[:,j,:], label="input")
        pl.plot(seq_target_data[:,j,:], label="target")
        pl.plot(outs[:,j,:], label="pred")
    pl.legend()
    pl.subplot(312)
    pl.title("all outputs over sample iterations")
    allouts = np.asarray(allouts)
    pl.plot(allouts.flatten())

    pl.subplot(313)
    pl.title("cost")
    pl.plot(allcosts)
    
    # pl.plot(outs)
    pl.show()

        
# eval: free running
def sample_fr(args):
    data_pointer = 0
    data = gen_data(args.seq_length * args.train_steps + 1)
    
    model = Model(n_steps = 1, batch_size = 1)
    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    # with tf.device("/gpu:1"):
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    saver = tf.train.Saver(tf.all_variables(), max_to_keep = 100)

    saver.restore(session, "recurrent_network_1b.ckpt")
    
    # eval
    prev_state = session.run(tf.random_uniform([1, model.cell.state_size], -1., 1.))
    data_pointer = 0
    allouts  = []
    allcosts = []
    params = {

            "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "seq_width": model.seq_width
        }
    seq_input_data  = get_seq_input_data(data_pointer, data, params)
    print seq_input_data.shape
    seq_input_data  = np.random.rand(model.n_steps, model.batch_size, model.seq_width) * 0.01
    print seq_input_data.shape, seq_input_data
    seq_target_data = get_seq_input_data(data_pointer+1, data, params)
    for i in range(args.train_steps):
        # print type(seq_target_data)
        data_pointer += model.n_steps
        # print "pstate", prev_state
        feed = {model.early_stop: model.n_steps,
                model.seq_input: seq_input_data,
                model.initial_state: prev_state}
        # feed = {early_stop:n_steps, seq_input: seq_input_data}
        # define our feeds. 
        # early_stop can be varied, but seq_input needs to match the shape that was defined earlier

        outs, fstate = session.run([model.output2, model.final_state], feed_dict=feed)
        # outs +=  np.random.normal(0., 0.05, outs.shape)
        # outs, fstate, tcost, opt = session.run([output, final_state, cost, optimizer], feed_dict=feed)
        if np.random.rand() < 0.01:
            seq_input_data  = get_seq_input_data(data_pointer, data, params)
        else:
            seq_input_data  = outs
        prev_state = fstate
        # print "fstate", fstate,tcost
        allouts.append(outs)
        # allcosts.append(tcost)
        # tcost = session.run(cost, feed_d)
        if i % 10 == 0:
            # print "cost[%d] = %f" % (i, tcost)
            print "iter %d" % i
    
    # run once
    # output is a list, each item being a single timestep. Items at t>early_stop are all 0s
    # print outs
    print type(outs)
    print len(outs)
    print type(outs[0])
    print outs[0].shape
    print "allouts", len(allouts)

    pl.subplot(311)
    params = {
        "n_steps": args.train_steps,
        "batch_size": model.batch_size,
        "seq_width": model.seq_width
        }
    seq_target_data = get_seq_input_data(0, data, params)
    pl.plot(seq_target_data[:,0,:])
    pl.subplot(312)
    allouts = np.asarray(allouts)
    pl.plot(allouts.flatten())

    pl.subplot(313)
    pl.plot(allcosts)
    
    # pl.plot(outs)
    pl.show()

def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    elif args.mode == "sample_fr":
        sample_fr(args)
    else:
        print("unknown mode")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train", help="Either train, sample, sample_fr")
    parser.add_argument("--seq_length", type=int, default=10, help="seq length")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--train_steps", type=int, default=100, help="number of training iterations")

    args = parser.parse_args()
    main(args)
