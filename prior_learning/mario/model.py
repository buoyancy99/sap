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

def make_model(blocks, actions, seq_mask, action_space = 12, seq_len = 128, block_size = 12):
    # blocks_input = tf.placeholder(tf.float32, [None, seq_len, 8, block_size, block_size])
    # actions_input = tf.placeholder(tf.int64, [None, seq_len, ])
    batch_size = tf.cast(tf.shape(blocks)[0], tf.int64)
    blocks = tf.reshape(blocks, (batch_size * seq_len * 8, block_size, block_size, 1))

    pos_idx = tf.tile(tf.range(8, dtype=tf.int64), (batch_size * seq_len,))
    pos_idx = tf.reshape(pos_idx, (batch_size * seq_len * 8, 1))

    actions = tf.reshape(actions, (batch_size * seq_len, 1))
    action_idx = tf.tile(actions, (1, 8))
    action_idx = tf.reshape(action_idx, (batch_size * seq_len * 8, 1))

    idx = tf.range(batch_size * seq_len * 8, dtype=tf.int64)
    idx = tf.reshape(idx, (batch_size * seq_len * 8, 1))

    mask_idx = tf.concat([idx, pos_idx, action_idx], 1)
    mask_value = tf.ones((batch_size * seq_len * 8, ))

    mask = tf.sparse.SparseTensor(mask_idx, mask_value, tf.convert_to_tensor((batch_size * seq_len * 8, 8, action_space), dtype=tf.int64))

    # X = layers.Conv2D(16, 3, 2, "same", activation=tf.nn.relu)(blocks)
    # X = layers.Flatten(input_shape=(16, 3, 3))(X)
    # X = layers.Dense(128, activation=tf.nn.relu)(X)
    # X = layers.Dense(8 * action_space)(X)

    X = conv2d(1, 8, 3, 2, "SAME", blocks, "conv1")
    X = conv2d(8, 16, 3, 2, "SAME", X, "conv2")
    X = tf.reshape(X, (-1, 16 * 3 * 3))
    X = fc(16 * 3 * 3, 128, X, "fc1")
    X = fc(128, 8 * action_space, X, "fc2", None)

    X = tf.reshape(X, (batch_size * seq_len * 8, 8, action_space))
    reg = tf.reduce_sum(tf.reshape(tf.abs(X), (batch_size,  seq_len * 8 * 8 * action_space)), [1])
    X = X * tf.sparse.to_dense(mask)
    X = tf.reshape(X, (batch_size, seq_len, 8, 8, action_space))
    output = tf.reduce_sum(X, [2, 3, 4]) * seq_mask
    output = tf.reduce_sum(output, [1])

    return output, reg


