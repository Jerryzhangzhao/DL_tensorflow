import tensorflow as tf


# convolution
k = tf.constant([
    [1, 0, 1],
    [2, 1, 0],
    [0, 0, 1]
], dtype=tf.float32, name='k')

k1 = tf.constant([[
    [1, 0, 1],
    [2, 1, 0],
    [0, 0, 1]
],[
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1]
]
], dtype=tf.float32, name='k1')


i = tf.constant([
    [4, 3, 1, 0],
    [2, 1, 0, 1],
    [1, 2, 4, 1],
    [3, 1, 0, 2]
], dtype=tf.float32, name='i')
kernel = tf.reshape(k1, [3, 3, 1, 2], name='kernel')
image = tf.reshape(i, [1, 4, 4, 1], name='image')

res = tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID"))

t = k[:,0]

# VALID means no padding
with tf.Session() as sess:
    print(sess.run(res))
    print(sess.run(t))