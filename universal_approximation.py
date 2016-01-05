import numpy as np
import tensorflow as tf
import matplotlib.pylab as pl


function_to_learn = lambda x: np.sin(x) + 0.1*np.random.randn(*x.shape)

NUM_HIDDEN_NODES = 100
NUM_EXAMPLES = 1000
TRAIN_SPLIT = .8
MINI_BATCH_SIZE = 100
NUM_EPOCHS = 10000

# NUM_EXAMPLES = 1000
# TRAIN_SPLIT = 0.8

np.random.seed(1000) # for reproducibility 
all_x = np.float32(
    np.random.uniform(-2*np.pi, 2*np.pi, (1, NUM_EXAMPLES))).T
np.random.shuffle(all_x)
train_size = int(NUM_EXAMPLES*TRAIN_SPLIT)
trainx = all_x[:train_size]
validx = all_x[train_size:]
trainy = function_to_learn(trainx)
validy = function_to_learn(validx)

pl.plot(trainx, trainy, "ro", alpha=0.3, label="train")
pl.plot(validx, validy, "go", alpha=0.3, label="valid")
pl.legend()
pl.show()

X = tf.placeholder(tf.float32, [None, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

# NUM_HIDDEN_NODES = 10


def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def model(X, num_hidden=10):    
    w_h = init_weights([1, num_hidden], 'xavier', xavier_params=(1, num_hidden))
    b_h = init_weights([1, num_hidden], 'zeros')
    h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
     
    w_o = init_weights([num_hidden, 1], 'xavier', xavier_params=(num_hidden, 1))
    b_o = init_weights([1, 1], 'zeros')
    return tf.matmul(h, w_o) + b_o

yhat = model(X, NUM_HIDDEN_NODES)
    
train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(yhat - Y))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

errors = []
for i in range(NUM_EPOCHS):
    for start, end in zip(range(0, len(trainx), MINI_BATCH_SIZE), range(MINI_BATCH_SIZE, len(trainx), MINI_BATCH_SIZE)):
        sess.run(train_op, feed_dict={X: trainx[start:end], Y: trainy[start:end]})
    mse = sess.run(tf.nn.l2_loss(yhat - validy),  feed_dict={X:validx})
    errors.append(mse)
    if i%100 == 0: print "epoch %d, validation MSE %g" % (i, mse)
pl.plot(errors)
pl.xlabel('#epochs')
pl.ylabel('MSE')
pl.show()

# predict

TEST_SIZE = 1000
testx = np.float32(np.random.uniform(-4*np.pi, 4*np.pi, (1, TEST_SIZE))).T

testy = sess.run(yhat, feed_dict={X: testx})
pl.plot(trainx, trainy, "ro", alpha=0.3, label="train")
pl.plot(validx, validy, "go", alpha=0.3, label="valid")
pl.plot(testx, testy, "bo", alpha=0.3, label="test")
pl.legend()
pl.show()
