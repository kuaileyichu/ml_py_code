import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
 
 
def LeNet(x):
    # Arguments used for tf.truncated_normal , randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
 
    '''
    Layer 1
    '''
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
    '''
    Layer 2
    '''
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
    '''
    full connection
    '''
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400. 压平
    fc0 = flatten(conv2)
 
    '''
    fc1
    '''
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)
 
    '''
    fc2
    '''
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
 
    '''
    fc3
    '''
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

# 通过TensorFlow-Slim来定义LeNet-5的网络结构。
def lenet5(inputs):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])# 看面试图片尺寸
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 500, scope='layer5')
    net = slim.fully_connected(net, 10, scope='output')
    return net

def train(mnist):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    y = lenet5(x)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(3000):
            xs, ys = mnist.train.next_batch(100)
            _, loss_value = sess.run([train_op, loss], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))

def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
