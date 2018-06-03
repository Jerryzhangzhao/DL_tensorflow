import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 批次大小
batch_size = 100

# 批次数目
m_batch = mnist.train.num_examples//batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x, w)+b)

# 二次代价函数
#loss = tf.reduce_mean(tf.square(prediction - y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化
init = tf.global_variables_initializer()

# argmax 输出维度上的最大值的index
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(41):
        for batch in range(m_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("iter "+ str(epoch) + ", test accuracy: " + str(acc) )










