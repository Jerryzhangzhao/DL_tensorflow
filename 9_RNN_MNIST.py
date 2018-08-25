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


def rnn(x, weights, biases):
    inputs = tf.reshape(x, [-1, max_time, n_inputs])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    # 'state' is a tensor of shape[batch_size, cell_state_size]
    # state[0]:cell state,state[1]:hidden state
    # state is final state of the time serials while output contains all the states of each time point
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

    results = tf.nn.softmax(tf.matmul(state[1], weights) + biases)
    return results


# state is LSTMStateTuple, state[0]:cell state,state[1]:hidden state
''' 
LSTMStateTuple(c=array([[ 1.4003208 ,  5.3911433 ,  1.3681278 , ...,  0.88553107,
         2.6449218 ,  3.021435  ],
       [ 0.73893404,  7.522912  ,  5.368811  , ...,  8.097184  ,
         1.5976303 , -0.4217282 ],
       [ 0.923707  ,  1.8645589 ,  2.7729654 , ..., -2.3037126 ,
         3.0440154 , -1.1315142 ],
       ...,
       [-2.4747496 ,  5.387638  , -1.5895548 , ...,  3.225986  ,
         2.19178   , -3.2867982 ],
       [-2.6102498 ,  6.910054  , -0.3397467 , ...,  5.625205  ,
         0.63867795, -2.3031251 ],
       [-3.755093  ,  7.8372283 ,  4.604886  , ...,  3.7100544 ,
         0.19672015, -0.41049248]], dtype=float32), 
         
       h=array([[ 0.5207462 ,  0.7044978 ,  0.79254985, ...,  0.6382765 ,
         0.87966275,  0.9602473 ],
       [ 0.5697053 ,  0.90182847,  0.9575436 , ...,  0.9356195 ,
         0.83545005, -0.38531256],
       [ 0.3323384 ,  0.7125735 ,  0.8852245 , ..., -0.69027716,
         0.8095767 , -0.6152911 ],
       ...,
       [-0.8340237 ,  0.7708159 , -0.8142196 , ...,  0.68907934,
         0.86848384, -0.91779894],
       [-0.9046849 ,  0.9284657 , -0.3011895 , ...,  0.7684504 ,
         0.4953476 , -0.9350287 ],
       [-0.84070975,  0.836363  ,  0.819017  , ...,  0.7208597 ,
         0.17305236, -0.31775635]], dtype=float32))
'''

# prediction
prediction = rnn(x, weights, biases)

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
