import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 批次大小
batch_size = 100

# 批次数目
m_batch = mnist.train.num_examples//batch_size
keep_prob = tf.placeholder(tf.float32)

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

lr = tf.Variable(0.001, dtype=tf.float32)

# 创建一个简单的神经网络
w1 = tf.Variable(tf.truncated_normal([784, 500]))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.relu(tf.matmul(x, w1)+b1)
L1_dropout = tf.nn.dropout(L1, keep_prob)

w2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.relu(tf.matmul(L1_dropout,w2)+b2)
L2_dropout = tf.nn.dropout(L2, keep_prob)


w3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_dropout, w3)+b3)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(prediction - y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化
init = tf.global_variables_initializer()

# argmax 输出维度上的最大值的index
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

acc_step_list = []
test_acc_list = []
train_acc_list = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        sess.run(tf.assign(lr, 0.001*(0.9**epoch)))
        for batch in range(m_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})

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









