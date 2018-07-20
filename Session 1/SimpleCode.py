import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # Fake data
    x = np.random.rand(100).astype(np.float32)
    y = x * 0.1 + 0.3

    # Create tensorflow graph
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))

    _y = W * x + b

    loss = tf.reduce_mean(tf.square(_y - y))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print("{}, W={}, b={}".format(step, sess.run(W), sess.run(b)))