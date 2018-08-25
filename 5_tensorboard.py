'''
1.tensorborad 网络结构可视化

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 批次大小
batch_size = 100


# 参数概要
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean) # 平均值
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev) # 标准差
        tf.summary.scalar("max", tf.reduce_max(var)) # 最大值
        tf.summary.scalar("min", tf.reduce_min(var)) # 最小值
        tf.summary.histogram("histogram", var) # 直方图


# 批次数目
m_batch = mnist.train.num_examples//batch_size
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("input"):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    y = tf.placeholder(tf.float32, [None, 10], name="input_y")

lr = tf.Variable(0.001, dtype=tf.float32)

# 创建一个简单的神经网络
with tf.name_scope("layers"):
    with tf.name_scope("layer1"):
        w1 = tf.Variable(tf.truncated_normal([784, 500]), name="layer1_w")
        variable_summaries(w1)
        b1 = tf.Variable(tf.zeros([500])+0.1, name="layer1_b")
        variable_summaries(b1)
        L1 = tf.nn.relu(tf.matmul(x, w1)+b1, name="layer1_relu")
        L1_dropout = tf.nn.dropout(L1, keep_prob, name="layer1_dropout")

    with tf.name_scope("layer2"):
        w2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name="layer2_w")
        b2 = tf.Variable(tf.zeros([300])+0.1, name="layer2_b")
        L2 = tf.nn.relu(tf.matmul(L1_dropout,w2)+b2, name="layer2_relu")
        L2_dropout = tf.nn.dropout(L2, keep_prob, name="layer2_dropout")

    with tf.name_scope("layer3"):
        w3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name="layer3_w")
        b3 = tf.Variable(tf.zeros([10])+0.1, name="layer3_b")
        prediction = tf.nn.softmax(tf.matmul(L2_dropout, w3)+b3, name="layer3_prediction")

with tf.name_scope("loss"):
    # 二次代价函数
    # loss = tf.reduce_mean(tf.square(prediction - y))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    # 使用梯度下降
    # train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope("init"):
    # 初始化
    init = tf.global_variables_initializer()

with tf.name_scope("accuracy"):
    # argmax 输出维度上的最大值的index
    # 结果存放在一个布尔型列表中
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 求准确率
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

acc_step_list = []
test_acc_list = []
train_acc_list = []

# 合并所有的summary
merger = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter("log/", sess.graph)

    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001*(0.9**epoch)))
        for batch in range(m_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
            summary, _ = sess.run([merger, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        writer.add_summary(summary, epoch)

        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})

        acc_step_list.append(epoch)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        print("iter " + str(epoch) + ", test accuracy: " + str(test_acc) + ", train accuracy: " + str(train_acc)
              + ", lr: " + str(sess.run(lr)))

    plt.plot(acc_step_list, train_acc_list)
    plt.plot(acc_step_list, test_acc_list)
    plt.show()








