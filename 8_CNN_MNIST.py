import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# dataSet
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# batch num and batch size
batch_size = 100
batch_num = mnist.train.num_examples // batch_size
print("batch_num: "+str(batch_num))

# network construction
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

x_reshape = tf.reshape(x, [-1, 28, 28, 1])

w1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, tf.float32, shape=[32]))

x1_conv = tf.nn.relu(tf.nn.conv2d(x_reshape, w1, [1, 1, 1, 1], padding="SAME") + b1)
x1_pool = tf.nn.max_pool(x1_conv, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", data_format="NHWC")

w2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, tf.float32, shape=[64]))

x2_conv = tf.nn.relu(tf.nn.conv2d(x1_pool, w2, [1, 1, 1, 1], padding="SAME") + b2)
x2_pool = tf.nn.max_pool(x2_conv, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", data_format="NHWC")

wfc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1))
bfc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

x2_conv_flat = tf.reshape(x2_pool, shape=[-1, 7 * 7 * 64])

x3_fc = tf.nn.relu(tf.matmul(x2_conv_flat, wfc1) + bfc1)

wfc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
bfc2 = tf.Variable(tf.constant(0.1, tf.float32, shape=[10]))

prediction = tf.nn.relu(tf.matmul(x3_fc, wfc2) + bfc2)

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# optimizer
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# prediction accuracy calculation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(batch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + " acc: " + str(acc))
