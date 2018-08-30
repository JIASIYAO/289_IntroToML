import time

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def optimize(x, y, pred, loss, optimizer, training_epochs, batch_size):
    acc = []
    with tf.Session() as sess:  # start training
        sess.run(tf.global_variables_initializer())  # Run the initializer
        for epoch in range(training_epochs):  # Training cycle
            avg_loss = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_loss += c / total_batch

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = accuracy_.eval({x: mnist.test.images, y: mnist.test.labels})
            acc.append(accuracy)
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss),
                  "accuracy={:.9f}".format(accuracy))
    return acc


def train_linear(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean((y - pred)**2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def train_logistic(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # YOUR CODE HERE
    pred = 0
    loss = 0
    ################
    z = tf.matmul(x, W) + b
    z_max = tf.reduce_max(z)
    sm_sum = tf.reduce_sum(tf.exp(z - z_max), axis=0)
    pred = tf.exp(z - z_max)/sm_sum
    loss = (-1.0) * tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(pred, y))))


    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def train_nn(learning_rate=0.01, training_epochs=50, batch_size=50, n_hidden=128):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([784, n_hidden]))
    W2 = tf.Variable(tf.random_normal([n_hidden, 10]))
    b1 = tf.Variable(tf.random_normal([n_hidden]))
    b2 = tf.Variable(tf.random_normal([10]))

    # YOUR CODE HERE
    pred = 0
    loss = 0
    ################
    z1 = tf.matmul(x, W1) + b1
    z2 = tf.matmul(tf.tanh(z1), W2) + b2
    z2_max = tf.reduce_max(z2)
    sm_sum = tf.reduce_sum(tf.exp(z2 - z2_max), axis=0)
    pred = tf.exp(z2 - z2_max)/sm_sum
    loss = (-1.0) * tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(pred, y))))


    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


def main():
    #for batch_size in [50, 100, 200]:
    #    time_start = time.time()
    #    acc_linear = train_linear(batch_size=batch_size)
    #    print("train_linear finishes in %.3fs" % (time.time() - time_start))

    #    plt.plot(acc_linear, label="linear bs=%d" % batch_size)
    #    plt.legend()
    #    plt.savefig('linear_%d.png'%batch_size, format='png')

    #acc_logistic = train_logistic()
    #plt.plot(acc_logistic, label="logistic regression")
    #plt.legend()
    #plt.show()

    acc_nn = train_nn()
    plt.plot(acc_nn, label="neural network")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tf.set_random_seed(0)
    main()
