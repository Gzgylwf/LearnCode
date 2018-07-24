import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def course1():
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


def course2():
    m1 = tf.constant([[3, 4]])
    m2 = tf.constant([[2],[2]])

    product = tf.matmul(m1, m2)

    with tf.Session() as sess:
        result = sess.run(product)
        print(result)
        sess.close()


def course3():
    state = tf.Variable(0, name='counter')
    one = tf.constant(1)

    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # If graph include any Variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(10):
            sess.run(update)
            print(sess.run(state))
        sess.close()


def course4():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1:[7.], input2:[5.]}))
        sess.close()


def add_layer(input, in_size, out_size, activation_function=None, namespace="Unkown"):
    with tf.name_scope(namespace):
        W = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights')
        tf.summary.histogram("{}::{}".format(namespace, "Weights"), W)
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='Bias')
        tf.summary.histogram("{}::{}".format(namespace, "Bias"), b)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input, W) + b

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram("{}::{}".format(namespace, "Output"), outputs)
    return outputs


def course5():
    # Fake data
    x = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.1, x.shape)
    y = np.square(x) - 0.8 + noise

    # Input
    xi = tf.placeholder(tf.float32, [None, 1])
    yi = tf.placeholder(tf.float32, [None, 1])

    # Build
    l1 = add_layer(xi, 1, 10, activation_function=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(yi - prediction), 1))

    train = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
    init = tf.global_variables_initializer()

    # Run
    with tf.Session() as sess:
        sess.run(init)

        # Visulation
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y)
        plt.ion()

        feed_input = {xi:x, yi:y}
        for i in range(1000):
            sess.run(train, feed_dict=feed_input)
            if i % 50 == 0:
                #print(sess.run(loss, feed_dict=feed_input))
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                plt_predict_y = (sess.run(prediction, feed_dict=feed_input))
                lines = ax.plot(x, plt_predict_y, 'r-', lw=5)
                plt.pause(0.1)
        sess.close()


def course6():
    # Fake data
    x = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.1, x.shape)
    y = np.square(x) - 0.8 + noise

    # Input
    with tf.name_scope('inputs'):
        xi = tf.placeholder(tf.float32, [None, 1], name='x_input')
        yi = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # Layer
    layer = add_layer(xi, 1, 100, activation_function=tf.nn.relu, namespace="Layer0")
    layer_1 = add_layer(layer, 100, 10, activation_function=tf.nn.relu, namespace="Layer1")
    prediction = add_layer(layer_1, 10, 1, activation_function=tf.nn.relu, namespace="Output_Layer")

    # Build
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(yi - prediction), 1))
        tf.summary.scalar("loss", loss)

    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
    
    init = tf.global_variables_initializer()

    # Run
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log", sess.graph)
        sess.run(init)
        feed_input = {xi:x, yi:y}
        for i in range(1000):
            sess.run(train, feed_dict=feed_input)
            if i % 50 == 0:
                print(sess.run(loss, feed_dict=feed_input))
                result = sess.run(merged, feed_dict=feed_input)
                writer.add_summary(result, i)
        
        sess.close()


if __name__ == '__main__':
    # Basic regress
    #course1()

    # Session control
    #course2()

    # Variable
    #course3()

    # Placeholder
    #course4()

    # Build a NN, active func/ layer, input/ optimizer
    #course5()

    # Tensorboard
    course6()
    