import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.set_random_seed(0)

hello = tf.constant("hello ")
world = tf.constant("world!")
print(hello)
print(type(hello))

with tf.Session() as sess: # no need to close it after use
    result = sess.run(hello+world)
    
print(result)

a = tf.constant(10)
b = tf.constant(20)

with tf.Session() as sess: # no need to close it after use
    result = sess.run(a+b)
    
print(result)

const = tf.constant(10)
fill_mat = tf.fill((4,4),10) # fill a tensor with scalar value, size, number
myzeros = tf.zeros(4,4)
myones = tf.ones(4,4)
myrandn = tf.random_normal((4,4), 0.0, 1.0) # size, mean, std
myrandu = tf.random_uniform((4,4), 0.0, 1.0) # size, min, max

my_ops = [const, fill_mat, myzeros, myrandn, myrandu]

with tf.Session() as sess:
    for op in my_ops:
        print(sess.run(op))
        print("-"*30)
        
print(myrandn.get_shape())

result = tf.matmul(myrandn,myrandu)

with tf.Session() as sess:
    print(sess.run(result))
    
# 2 main types of objects:
    # variables: parameters to be tuned during the optimization to fit the model,
    #            holds biases and weights, need to be initialized
    # placeholders: initially empty, to be fed with input data (type + shape)
   
my_var = tf.Variable(initial_value = myrandu)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(my_var))
    
ph = tf.placeholder(tf.float32, shape = (None,4)) # dtype, shape (None for auto)

rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a + b
mul_op = a * b #(tf.matmul(a,b))

#tf.multiply() # element-wise
with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict = {a:rand_a, b:rand_b})
    print(add_result)
    mul_result = sess.run(mul_op, feed_dict = {a:rand_a, b:rand_b})
    print(mul_result)

