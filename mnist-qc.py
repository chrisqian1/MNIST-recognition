#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#Step 1
    #定义算法公式Softmax Regression
x=tf.placeholder(tf.float32,[None,784])
W1=tf.Variable(tf.zeros([784,400]))
b1=tf.Variable(tf.zeros([400]))
a=tf.matmul(x,W1)+b1
W2=tf.Variable(tf.zeros([400,10]))
b2=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(a,W2)+b2)

#Step 2S
    #定义损失函数，选定优化器，并指定优化器优化损失函数
y_=tf.placeholder('float',[None,10])

# 交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用梯度下降法（0.01的学习率）来最小化这个交叉熵损失函数
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()


#Step 3    
#使用随机梯度下降训练数据
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(10000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%100==0:
        print(sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys}))
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))         #tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真值
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      #用平均值来统计测试准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  #打印测试信息
sess.close()

