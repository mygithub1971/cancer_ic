# example nn, y = Wx + b
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)
#tf.set_random_seed(0)

n_features = 10
n_dense_neurons = 3
x = tf.placeholder(tf.float32, (None, n_features)) # rows auto
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x, W)
z = tf.add(xW, b)

a = tf.nn.sigmoid(z)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x:np.random.normal(0.0,1.0,(1,n_features))}) # feed cannot be tf
    #layer_out = sess.run(a) # must feed ph

print(layer_out)

x_data = np.linspace(0,10,11) + np.random.uniform(-1.0, 1.0, 11)
y_label = np.linspace(0,10,11) + np.random.uniform(-1.0, 1.0, 11)

plt.plot(x_data, y_label, "*")

m = tf.Variable(np.random.rand())
b = tf.Variable(np.random.rand())

error = 0

for x, y in zip(x_data, y_label):
    y_pred = m*x + b
    error += (y - y_pred)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 10
    for i in range(training_steps):
        sess.run(train)
        print(sess.run([m,b, error]))
    mm, bb = sess.run([m,b])
    
xx = np.linspace(-1,11,100)

yy = mm*xx + bb

plt.plot(xx, yy, 'r')
plt.plot(x_data, y_label, '*')




