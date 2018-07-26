import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, sent_length, nclass, vocab_size, embedding_size, filter_sizes, num_filters):
        # Define I/O and dropout
        self.x = tf.placeholder(tf.int32, [None, sent_length], name='Input')
        self.y = tf.placeholder(tf.float32, [None, nclass], name='Output')
        self.dropout = tf.placeholder(tf.float32, name='Dropout')

        # Embeding layer
        with tf.device('/cpu:0'), tf.name_scope('Embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_embedding')
            self.embbed_chars = tf.nn.embedding_lookup(W, self.x)
            self.embbed_chars_expanded = tf.expand_dims(self.embbed_chars, -1)

        # Pooling layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("Conv-Maxpool-{}".format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b'))
                conv = tf.nn.conv2d(self.embbed_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='Conv')
                # Activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='Relu')
                # Pooling
                pool = tf.nn.max_pool(h, ksize=[1, sent_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='Pool')
                pool_outputs.append(pool)

        # Combine pool features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout layer
        with tf.name_scope('Dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout)

        # Scores and Predictions
        with tf.name_scope('Output'):
            W = tf.Variable(tf.truncated_normal([num_filters_total, nclass], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[nclass]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='Scores')
            self.prediction = tf.argmax(self.scores, 1, name='Prediction')

        # Loss and Accuracy
        with tf.name_scope('Loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y)
            self.loss = tf.reduce_mean(loss)

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# Training Procedure
def main():
    with tf.Graph().as_default():
        sess_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=sess_conf)
        with sess.as_default():
            cnn = TextCNN(sent_length=x_train.shape[1], nclass=2, vocab_size=len(vocabulary), embedding_size=FLAGS.embedding_size, 
                        filter_sizes=map(int, FLAGS.filter_sizes.split(',')), num_filters=FLAGS.num_filters)

            # Optimizer
            global_steps = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_steps)

            # Summary
            out_dir = "log"
            print("Writing to {}\n".format(out_dir))
            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpointing
            checkpoint_dir = "checkpoints"
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # RUN!
            sess.run(tf.variables_initializer())


# Train step
def train_step(x_batch, y_batch):
    feed_dict = { cnn.x: x_batch, cnn.y: y_batch, cnn.dropout = FLAGS.dropout_keep_prob }



if __name__ == '__main__':
    main()