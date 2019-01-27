# regression tf
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0.0, 10.0, 100000)

y = 5.0*x + 0.5 + np.random.normal(0.0,1.0,x.shape[0])

x_df = pd.DataFrame(x, columns = ['x'])
y_df = pd.DataFrame(y, columns = ['y'])

df = pd.concat([x_df, y_df], axis = 1)

df.sample(n=250).plot(kind='scatter', x = 'x', y = 'y')

batch_size = 10

m = tf.Variable(np.random.normal())
b = tf.Variable(np.random.normal())

init = init = tf.global_variables_initializer()

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_pred = m*xph + b

loss = tf.reduce_sum(tf.square(yph - y_pred))