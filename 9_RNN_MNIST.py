import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_inputs = 28  # 输入的一行,28维的数据
max_time = 28  # 时间序列长度
lstm_size = 100  # 隐层单元数
n_classes = 10
batch_size = 50
batch_num = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


def RNN(x, weights, biases):
    inputs = tf.reshape(x, [-1, max_time, n_inputs])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

# prediction
prediction = RNN(x, weights, biases)

# cost fun
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

# train
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct prediction
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # 判断是否预测正确，boolean

# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # reduce_mean 计算准确度

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for batch in range(batch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("epoch: " + str(epoch) + "  acc: " + str(acc))
