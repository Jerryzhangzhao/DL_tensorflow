# save and restore model using tf.train.Saver()
# often used in training stage

import tensorflow as tf
import numpy as np


# save model using tf.train.Saver()
def save_model(model_path):
    x_input = tf.placeholder(tf.float32, shape=[7, 7], name="x_input")
    x_reshape = tf.reshape(x_input, shape=[-1, 7, 7, 1], name='x_reshape')
    w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 5], stddev=0.1), name='w')
    conv = tf.nn.conv2d(x_reshape, w, [1, 1, 1, 1], padding="SAME", name='conv')

    w1 = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1), name='w1')
    w2 = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1), name='w2')
    y = tf.matmul(w1, w2, name='y')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(conv, feed_dict={x_input: np.random.rand(7, 7)})
        saver.save(sess, model_path)


# restore model
# restore model in this case, it requires a session in which the graph was launched
# and the variable don't have to been initialized
def restore_model(model_path):
    x_input = tf.placeholder(tf.float32, shape=[7, 7], name="x_input")
    x_reshape = tf.reshape(x_input, shape=[-1, 7, 7, 1], name='x_reshape')
    w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 5], stddev=0.1), name='w')
    conv = tf.nn.conv2d(x_reshape, w, [1, 1, 1, 1], padding="SAME", name='conv')

    w1 = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1), name='w1')
    w2 = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1), name='w2')
    y = tf.matmul(w1, w2, name='y')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        x_data = np.random.rand(7, 7)
        saver.restore(sess, model_path)  # here need the graph been defined in code
        conv_r = sess.run('conv:0', feed_dict={'x_input:0': x_data})
        print(conv_r)


# restore model using import_mata_graph
# by using import_meta_graph, we don't have to redefine the model again
def restore_model_with_mata_graph(model_path):
    with tf.Session() as sess:
        x_data = np.full((7, 7), 10)
        # import meta graph and restore variables
        saver = tf.train.import_meta_graph(_model_path + '.meta')
        saver.restore(sess, model_path)
        # get graph
        graph = tf.get_default_graph()

        # in this case,we need to know the tensor name define in the graph during training
        # and using get_tensor_by_name here
        conv_tensor = graph.get_tensor_by_name('conv:0')
        input_tensor = graph.get_tensor_by_name('x_input:0')

        conv_r = sess.run(conv_tensor, feed_dict={input_tensor: x_data})
        print(conv_r)


if __name__ == '__main__':
    _model_path = './saved_model_ckpt/model.ckpt'
    # 'model.ckpt' is the prefix of the model file,not the full name of the file stored in disk

    mode = 'mode_restore_with_meta_graph'  # 'mode_save' or 'mode_restore' or 'mode_restore_with_meta_graph'
    if mode == 'mode_save':
        save_model(_model_path)
    if mode == 'mode_restore':
        restore_model(_model_path)
    if mode == 'mode_restore_with_meta_graph':
        restore_model_with_mata_graph(_model_path)
