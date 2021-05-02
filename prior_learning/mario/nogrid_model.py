import tensorflow as tf

def conv2d(In_C, Out_C, k, s, p, X, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('w', [k, k, In_C, Out_C])
        b = tf.get_variable('b', [Out_C])
        return tf.nn.relu(tf.nn.conv2d(X, filter=W, strides=[1, s, s, 1], padding=p) + b)

def fc(In_C, Out_C, X, name, activation = tf.nn.relu):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('w', [In_C, Out_C])
        b = tf.get_variable('b', [Out_C])
        if activation is not None:
            return activation(tf.matmul(X, W) + b)
        else:
            return tf.matmul(X, W) + b

def make_model(observations, actions, seq_mask, action_space = 5, seq_len = 128):
    # blocks_input = tf.placeholder(tf.float32, [None, seq_len, 30, 30])
    # actions_input = tf.placeholder(tf.int64, [None, seq_len, ])
    batch_size = tf.cast(tf.shape(observations)[0], tf.int64)
    observations = tf.reshape(observations, (batch_size * seq_len, 30, 30, 1))

    action_idx = tf.reshape(actions, (batch_size * seq_len, 1))

    idx = tf.range(batch_size * seq_len, dtype=tf.int64)
    idx = tf.reshape(idx, (batch_size * seq_len, 1))

    mask_idx = tf.concat([idx, action_idx], 1)
    mask_value = tf.ones((batch_size * seq_len, ))

    mask = tf.sparse.SparseTensor(mask_idx, mask_value, tf.convert_to_tensor((batch_size * seq_len, action_space), dtype=tf.int64))

    X = conv2d(1, 4, 3, 2, "SAME", observations, "conv1")
    X = conv2d(4, 8, 3, 2, "SAME", X, "conv2")
    X = conv2d(8, 16, 3, 2, "SAME", X, "conv3")
    X = tf.reshape(X, (batch_size * seq_len, 16 * 4 * 4))
    X = fc(16 * 4 * 4, 64, X, "fc1")
    X = fc(64, action_space, X, "fc2", None)

    X = tf.reshape(X, (batch_size * seq_len, action_space))
    reg = tf.reduce_sum(tf.reshape(tf.abs(X), (batch_size,  seq_len *  action_space)), [1])
    X = X * tf.sparse.to_dense(mask)
    X = tf.reshape(X, (batch_size, seq_len, action_space))
    output = tf.reduce_sum(X, [2]) * seq_mask
    output = tf.reduce_sum(output, [1])

    return output, reg


