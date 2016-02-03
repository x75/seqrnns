import tensorflow as tf
import numpy as np

seqlen = 10
batchsize = 50
dim = 3

a = tf.placeholder(tf.float32, [seqlen, batchsize, dim])
b1 = tf.split(0, seqlen, a)
b2 = [tf.reshape(i, (batchsize, dim)) for i in tf.split(0, seqlen, a)]



with tf.device("/gpu:1"):
    result = a * 1

iop = tf.initialize_all_variables()
session = tf.Session()
session.run(iop)

feed = {a: np.random.rand(seqlen, batchsize, dim)}

session.run(result, feed)

print "a", a
print "b1", len(b1), type(b1), b1
print b1[0]
print b2[0]

