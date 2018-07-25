import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, sent_length, nclass, vocab_size, embedding_size, filter_sizes, num_filters):
        # Define I/O and dropout
        self.x = tf.placeholder(tf.int32, [None, sent_length], name='Input')
        self.y = tf.placeholder(tf.float32, [None, nclass], name='Output')
        self.dropout = tf.placeholder(tf.float32, name='Dropout')

        # Embeding layers
        