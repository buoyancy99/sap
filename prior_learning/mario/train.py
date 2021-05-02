import tensorflow as tf
from prior_learning.mario.dataloader import mario_dataset
from prior_learning.mario.model import make_model
from config.mario_config import config
from tensorboardX import SummaryWriter

block_size = config["block_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
trained_on = config["trained_on"]

writer = SummaryWriter('logs/prior_learning/mario/original')

dataset = mario_dataset()
blocks_input, actions_input, mask, reward = dataset.get_next()
reward_hat, reg_loss = make_model(blocks_input, actions_input, mask, 5, seq_len, block_size)
reg_loss = tf.reduce_mean(reg_loss) * 2e-5
loss = reg_loss + tf.losses.absolute_difference(reward, reward_hat, reduction=tf.losses.Reduction.MEAN) # pass the second value from iter.get_net() as label
learning_rate = tf.placeholder(tf.float32, [])
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(dataset.initializer)
    running_sum = 0
    running_reg = 0
    for i in range(20000):
        if i <= 999:
            lr = 1e-3
        elif i <= 4999:
            lr = 1e-4
        else:
            lr = 1e-5
            
        if i % 20 == 19:
            print(i + 1, running_sum / 100, running_reg / 100)
            writer.add_scalar('loss', running_sum / 100, i + 1)
            running_sum = 0
            running_reg = 0
        _, loss_value, reg_loss_value = sess.run([train_op, loss, reg_loss], feed_dict = {learning_rate: lr})
        running_sum += loss_value
        running_reg += reg_loss_value
        if i%5000 == 4999:
            save_path = saver.save(sess, "prior_learning/mario/ckpts/"+ "{}_reward_{}".format(i+1, trained_on))
            save_path = saver.save(sess, "prior_learning/mario/ckpts/" + "reward_{}".format(trained_on))
    print("Model saved in path: %s" % save_path)
