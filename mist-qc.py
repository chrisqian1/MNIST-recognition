#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
in_units=784
h1_units=300
x=tf.placeholder(tf.float32,[None,in_units])
w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
h1=tf.nn.relu(tf.matmul(x,w1)+b1)
w2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(h1,w2)+b2)

y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init=tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

for i in range(10000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%100==0:
        print(sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys}))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  #打印测试信息
sess.close()

