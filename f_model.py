import tensorflow as tf


class forward_model(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,filter_sizes, num_filters):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.learning_rate = tf.placeholder(tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        #W is set to not trainable to use pre-trained word embeddings, but it may not fit this dataset perfectly
        self.W = tf.Variable(
            tf.random_uniform([vocab_size, 50], -1.0, 1.0), trainable=False,
            name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Conv->Relu->Maxpool layers for all filters
        pool = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, 50, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            a = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            z = tf.nn.relu(tf.nn.bias_add(a, b))
            maxpool = tf.nn.max_pool(z, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                     padding='VALID')
            pool.append(maxpool)

        # Flatten the pooled matrix
        self.pooltogether = tf.concat(pool, 3)
        self.poolflattened = tf.reshape(self.pooltogether, [-1, len(filter_sizes)* num_filters])

        # Dropout to avoid overfitting
        self.pooldropout = tf.nn.dropout(self.poolflattened, self.dropout_keep_prob)

        # Make the predictions
        W = tf.get_variable("W", shape=[128 * 2, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        self.final = tf.nn.xw_plus_b(self.pooldropout, W, b)
        self.prediction = tf.argmax(self.final, 1)

        # Get the loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.final, labels=self.input_y))

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.input_y, 1)), "float"))
