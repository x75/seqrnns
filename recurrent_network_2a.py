# this is from https://github.com/yankev/tensorflow_example/blob/master/rnn_example.ipynb

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import matplotlib.pylab as pl


class TestStateSaver(object):
    def __init__(self, batch_size, state_size):
        self._batch_size = batch_size
        self._state_size = state_size
    def State(self, _):
        return tf.zeros(tf.pack([self._batch_size, self._state_size]))
    def SaveState(self, _, state):
        self.saved_state = state
        return tf.identity(state)

#Defining some hyper-params
num_units = 100       #this is the parameter for input_size in the basic LSTM cell
input_size = 1     #num_units and input_size will be the same
output_size = 1

batch_size = 10
seq_len = 30
num_epochs=100

def gen_data(min_length=50, max_length=55, n_batch=5):

    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    y = np.zeros((n_batch,))
    print X.shape, y.shape
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        #i changed this to a constant
        #length=55

        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/2-1), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    #X -= X.reshape(-1, 2).mean(axis=0)
    #y -= y.mean()
    return (X,y)

def gen_data2(k = 0, min_length=50, max_length=55, n_batch=5, freq = 2.):
    print "k", k
    # t = np.linspace(0, 2*np.pi, n_batch)
    t = np.linspace(k*n_batch, (k+1)*n_batch+1, n_batch+1, endpoint=False)
    # print "t.shape", t.shape, t, t[:-1], t[1:]
    # freq = 1.
    Xtmp = np.sin(t[:-1] * freq / (2*np.pi))
    print Xtmp.shape
    # Xtmp = [np.sin(t[i:i+max_length]) for i in range(n_batch)]
    # print len(Xtmp)
    X = np.array(Xtmp).reshape((n_batch, input_size))
    # X = 
    # y = np.zeros((n_batch,))
    y = np.sin(t[1:] * freq / (2 * np.pi)).reshape((n_batch, output_size))
    # print X,y
    # print X.shape, y.shape
    # for i in range(batch_size):
    #     pl.subplot(211)
    #     pl.plot(X[i,:,0])
    #     # pl.subplot(312)
    #     # pl.plot(X[i,:,1])
    # pl.subplot(212)
    # pl.plot(y)
    # pl.show()
    
    return (X,y)

def gen_data3(k = 0, seq_length = 300, seq_width = 1, batch_size = 10, freq = 1.):
    print "k", k

    # Xtmp = np.zeros((batch_size, seq_length, seq_width))

    X_batch = []
    y_batch = []
    # X_batch = np.zeros((batch_size, seq_length, seq_width))
    # y_batch = np.zeros((batch_size, seq_length, seq_width))

    # generate entire chunk of data
    t_start = seq_length * batch_size * k
    t_end   = seq_length * batch_size * (k + 1) + 1
    print "t_start", t_start, "t_end", t_end
    t       = np.linspace(t_start, t_end, seq_length * batch_size + 1, endpoint=False)
    sin     = np.sin(t * freq / (2 * np.pi))

    print "sin.shape", sin.shape
        
    for i in xrange(batch_size):
        # t = np.linspace(0, 2*np.pi, n_batch)
        # print "t.shape", t.shape, t, t[:-1], t[1:]
        # freq = 1.
        
        # Xtmp = np.sin(t[:-1] * freq / (2*np.pi))
        # print Xtmp.shape
        
        # Xtmp = [np.sin(t[i:i+max_length]) for i in range(n_batch)]
        # print len(Xtmp)
        x_start = (i * seq_length)
        x_end   = ((i + 1) * seq_length)
        # X = np.array(Xtmp).reshape((n_batch, input_size))
        X_batch.append(sin[x_start:x_end].reshape((seq_length, seq_width)))
        y_batch.append(sin[x_start+1:x_end+1].reshape((seq_length, seq_width)))
        # y = np.sin(t[1:] * freq / (2 * np.pi)).reshape((n_batch, output_size))
        
    
    return (X_batch,y_batch)

# print X.shape
# print y.shape

# check data
for k in range(3):
    X,y = gen_data3(k = k, seq_length = seq_len, seq_width = input_size, batch_size = batch_size, freq = 0.1)
    pl.subplot(3, 1, k + 1)
    for b in xrange(len(X)):
        print "b", b
        # t = np.linspace(k*batch_size, (k+1)*batch_size+1, batch_size+1, endpoint=False)
        # t = range(k*batch_size, (k+1)*batch_size)
        # print len(X)
        t = np.arange(b*seq_len, (b+1)*seq_len)
        print t.shape, X[b].shape
        print t.shape, y[b].shape
        pl.plot(t, X[b], "b-", label="x")
        pl.plot(t, y[b], "g-", label="y")
    pl.legend()
pl.show()

# sys.exit()





### Model Construction

cell = rnn_cell.BasicLSTMCell(num_units)    #we use the basic LSTM cell provided in TensorFlow
                                            #num units is the input-size for this cell

#create placeholders for X and y

# # from recurrent_network_1.py
# seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
# # sequence we will provide at runtime  
# early_stop = tf.placeholder(tf.int32)
# # what timestep we want to stop at
# inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]

inputs  = [tf.placeholder(tf.float32,shape=[batch_size, input_size]) for _ in range(seq_len)]
# inputs = [tf.placeholder(tf.float32, shape=[None, input_size])]
result = [tf.placeholder(tf.float32,shape=[batch_size, output_size]) for _ in range(seq_len)]
# result = tf.placeholder(tf.float32, shape=[batch_size, output_size])

outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)   #note that outputs is a list of seq_len
                                                            #each element is a tensor of size [batch_size,num_units]
# state_saver = TestStateSaver(batch_size, 2*num_units)
# outputs, states = rnn.state_saving_rnn(cell, inputs, state_saver=state_saver, state_name="lstm_state") # dtype=tf.float32)   #note that outputs is a list of seq_len
#                                                             #each element is a tensor of size [batch_size,num_units]

# outputs2 = outputs[-1]   #we actually only need the last output from the model, ie: last element of outputs


#We actually want the output to be size [batch_size, 1]
#So we will implement a linear layer to do this

W_o = tf.Variable(tf.random_normal([num_units,1], stddev=0.01))     
b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

# outputs2 = outputs[-1]

# outputs3 = tf.matmul(outputs2,W_o) + b_o
outputs3 = tf.matmul(outputs, W_o) + b_o

cost = tf.reduce_mean(tf.pow(outputs3-result,2))    #compute the cost for this batch of data
# cost = tf.reduce_mean(tf.nn.l2_loss(outputs3-result))    #compute the cost for this batch of data

#compute updates to parameters in order to minimize cost

#train_op = tf.train.GradientDescentOptimizer(0.008).minimize(cost)
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

### Generate Validation Data
X_val,y_val = gen_data3(k = 17, seq_length = seq_len, seq_width = input_size, batch_size = batch_size, freq = 0.1)
# gen_data3(17, 50,seq_len,batch_size)
# X_val = []
# for i in range(seq_len):
#     X_val.append(tempX[:,i,:])

# pl.plot(X_val)
# pl.plot(y_val)
# pl.show()

### Execute

pl.ion()
pl.figure()
pl.subplot(311)
# pl.show()

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model

    for k in range(num_epochs):

        #Generate Data for each epoch
        #What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
        #this is required to feed data into rnn.rnn
        tempX, y = gen_data3(k = k, seq_length = seq_len, seq_width = input_size, batch_size = batch_size, freq = 0.1) #
        print len(tempX)
        tempX = np.array(tempX)
        y = np.array(y)
        
        # print type(X)
        print tempX.shape, y.shape
        X = []
        for i in range(seq_len):
            X.append(tempX[:,i,:])
        # print "X, y", X.shape, y.shape
        #Create the dictionary of inputs to feed into sess.run
        temp_dict = {inputs[i]:X[i] for i in range(seq_len)}
        # temp_dict = {inputs[0]: X, result: y} # for i in range(seq_len)}
        temp_dict.update({result: y})

        _, out = sess.run([train_op, outputs3], feed_dict=temp_dict)   #perform an update on the parameters

        print "out", out.shape

        # val_dict = {inputs[i]:X_val[i] for i in range(seq_len)}  #create validation dictionary
        val_dict = {inputs[0]: X_val} # [i] for i in range(seq_len)}  #create validation dictionary
        val_dict.update({result: y_val})
        c_val = sess.run(cost, feed_dict = val_dict )            #compute the cost on the validation set

        # print "blub.shape", blub.shape
        if k % 10 == 0:
            blub = sess.run(outputs3, feed_dict = val_dict)
            pl.subplot(311)
            pl.plot(blub)
            pl.subplot(312)
            pl.plot(y)
            pl.draw()
        
        print "Validation cost: {}, on Epoch {}".format(c_val,k)

pl.subplot(313)
pl.plot(X_val, label="input")
pl.plot(blub, label="pred")
pl.plot(y_val, label="y")
pl.legend()
pl.ioff()
pl.show()
