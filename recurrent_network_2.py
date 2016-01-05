# from https://github.com/yankev/tensorflow_example/blob/master/rnn_example.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pylab as pl
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

#Defining some hyper-params
num_units = 2       #this is the parameter for input_size in the basic LSTM cell
input_size = 2      #num_units and input_size will be the same

batch_size = 50
seq_len = 55
num_epochs=100

# Now we will generate the data. This code was taken from the RNN example from the Lasagne. https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py

# Summary of what type of data we are generating:

# Generate a batch of sequences for the "add" task, e.g. the target for the
# following
# ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
#   |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``
# would be 0.3 + .9 = 1.2. 

# The input and the output of the function gen_data:

# Parameters
# ----------
# min_length : int
#     Minimum sequence length.
# max_length : int
#     Maximum sequence length.
# n_batch : int
#     Number of samples in the batch.
# Returns
# -------
# X : Input to the network, of shape (n_batch, max_length, 2), where the last
#     dimension corresponds to the two sequences shown above.
# y : Correct output for each sample, shape (n_batch,)

# In [ ]:

def gen_data(min_length=50, max_length=55, n_batch=5):

    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    y = np.zeros((n_batch,))
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

# In [ ]:

X,y = gen_data(50, seq_len, batch_size)

print X.shape
print y.shape

for i in range(X.shape[0]):
    pl.plot(X[i], linewidth=0.2)
pl.plot(y, label="target")
pl.legend()
pl.show()


### Model Construction

cell = rnn_cell.BasicLSTMCell(num_units)    #we use the basic LSTM cell provided in TensorFlow
                                            #num units is the input-size for this cell

#create placeholders for X and y

inputs = [tf.placeholder(tf.float32,shape=[batch_size,input_size]) for _ in range(seq_len)]
result = tf.placeholder(tf.float32, shape=[batch_size])

outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)   #note that outputs is a list of seq_len
                                                            #each element is a tensor of size [batch_size,num_units]

outputs2 = outputs[-1]   #we actually only need the last output from the model, ie: last element of outputs


#We actually want the output to be size [batch_size, 1]
#So we will implement a linear layer to do this

W_o = tf.Variable(tf.random_normal([2,1], stddev=0.01))     
b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

outputs2 = outputs[-1]

outputs3 = tf.matmul(outputs2,W_o) + b_o       

cost = tf.reduce_mean(tf.pow(outputs3-result,2))    #compute the cost for this batch of data

#compute updates to parameters in order to minimize cost

#train_op = tf.train.GradientDescentOptimizer(0.008).minimize(cost)
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost) 

# In [ ]:

### Generate Validation Data
tempX,y_val = gen_data(50,seq_len,batch_size)
X_val = []
for i in range(seq_len):
    X_val.append(tempX[:,i,:])

# In [ ]:

### Execute

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model

    for k in range(num_epochs):

        #Generate Data for each epoch
        #What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
        #this is required to feed data into rnn.rnn
        tempX,y = gen_data(50,seq_len,batch_size)
        X = []
        for i in range(seq_len):
            X.append(tempX[:,i,:])

        #Create the dictionary of inputs to feed into sess.run
        temp_dict = {inputs[i]:X[i] for i in range(seq_len)}
        temp_dict.update({result: y})

        sess.run(train_op,feed_dict=temp_dict)   #perform an update on the parameters

        val_dict = {inputs[i]:X_val[i] for i in range(seq_len)}  #create validation dictionary
        val_dict.update({result: y_val})
        c_val = sess.run(cost, feed_dict = val_dict )            #compute the cost on the validation set
        
        print "Validation cost: {}, on Epoch {}".format(c_val,k)

