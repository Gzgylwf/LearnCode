import tensorflow as tf
import numpy as np
import pre_process
from tensorflow.contrib import learn
import datetime
import time
import os


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "data/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "data/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


class TextCNN(object):
    def __init__(self, sent_length, nclass, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Define I/O and dropout
        self.x = tf.placeholder(tf.int32, [None, sent_length], name='Input')
        self.y = tf.placeholder(tf.float32, [None, nclass], name='Output')
        self.dropout = tf.placeholder(tf.float32, name='Dropout')

        l2_loss = tf.constant(0.0)

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
                pooled_outputs.append(pool)

        # Combine pool features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout layer
        with tf.name_scope('Dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout)

        # Scores and Predictions
        with tf.name_scope('Output'):
            W = tf.Variable(tf.truncated_normal([num_filters_total, nclass], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[nclass]), name='b')
            # L2
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='Scores')
            self.predictions = tf.argmax(self.scores, 1, name='Prediction')

        # Loss and Accuracy
        with tf.name_scope('Loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# Training Procedure
def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        sess_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=sess_conf)
        with sess.as_default():
            cnn = TextCNN(sent_length=x_train.shape[1], nclass=2, vocab_size=len(vocab_processor.vocabulary_), embedding_size=FLAGS.embedding_size, 
                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))), num_filters=FLAGS.num_filters, l2_reg_lambda=0.4)

            # Optimizer
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Summary
            out_dir = "log"
            print("Writing to {}\n".format(out_dir))
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

            # Checkpointing
            checkpoint_dir = "checkpoints"
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            vocab_processor.save("data/vocab")

            # Train step
            def train_step(x_batch, y_batch):
                feed_dict = { cnn.x: x_batch, cnn.y: y_batch, cnn.dropout: FLAGS.dropout_keep_prob }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # Dev step
            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                cnn.x: x_batch,
                cnn.y: y_batch,
                cnn.dropout: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # RUN!
            sess.run(tf.global_variables_initializer())
            batches = pre_process.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def preprocess():
    print("Loading data...")
    x_text, y = pre_process.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()