# Import APIs
import tensorflow as tf

# image -> [B, H, W, C (RGB, feature maps)], EEG -> [B, # of electrode channel, timepoints, feature map]

def block(input, num_filter, pool_type, pool_size, pool_stride, regularizer, initializer):

    hidden = tf.layers.conv2d(inputs=input, filters=num_filter, kernel_size=(1, 3), activation=tf.nn.relu,
                              padding="valid", kernel_regularizer=regularizer, kernel_initializer=initializer)
    hidden = tf.layers.batch_normalization(inputs=hidden)
    hidden = tf.layers.conv2d(inputs=hidden, filters=num_filter, kernel_size=(1, 3), activation=tf.nn.relu,
                              padding="valid", kernel_regularizer=regularizer, kernel_initializer=initializer)
    hidden = tf.layers.batch_normalization(inputs=hidden)
    hidden = tf.layers.conv2d(inputs=hidden, filters=num_filter, kernel_size=(1, 3), activation=tf.nn.relu,
                              padding="valid", kernel_regularizer=regularizer, kernel_initializer=initializer)
    hidden = tf.layers.batch_normalization(inputs=hidden)

    if pool_type == "max":
        hidden = tf.layers.max_pooling2d(inputs=hidden, pool_size=pool_size, strides=pool_stride)
    elif pool_type == "average":
        hidden = tf.layers.average_pooling2d(inputs=hidden, pool_size=pool_size, strides=pool_stride)
    # else: ValueError("We have only Max and Average pooling...")

    return hidden

def drowsynet(eeg, label, reuse=False):
    with tf.variable_scope("drowsynet", reuse=reuse):
        # if reuse == True:
        #     keep_prob = 1.0
        # else:
        #     keep_prob = 0.5

        # We used L1-L2 regularizer, Xavier Initializer for all convolutional layers.
        # TODO-HW: regularization, initializer
        regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.01, scale_l2=0.001)
        initializer = tf.contrib.layers.xavier_initializer()

        # Core block 1
        hidden = block(input=eeg, num_filter=16, pool_type="max", pool_size=(1, 2), pool_stride=(1, 2),
                       regularizer=regularizer, initializer=initializer)

        # Core block 2
        hidden = block(input=hidden, num_filter=32, pool_type="max", pool_size=(1, 2), pool_stride=(1, 2),
                       regularizer=regularizer, initializer=initializer)

        # Core block 3
        hidden = block(input=hidden, num_filter=64, pool_type="average", pool_size=(1, 7), pool_stride=(1, 7),
                       regularizer=regularizer, initializer=initializer)

        # Classification
        hidden = tf.layers.flatten(inputs=hidden)
        hidden = tf.layers.dense(inputs=hidden, units=50, activation=tf.nn.leaky_relu)
        logit = tf.layers.dense(inputs=hidden, units=1, activation=None)
        logit = tf.squeeze(logit)

        # TODO-HW sparse_softmax_cross_entropy
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
        prediction = tf.math.round(tf.math.sigmoid(logit))
    return loss, prediction
