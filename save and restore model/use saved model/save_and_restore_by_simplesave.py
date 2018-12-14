# save and restore model using tf.saved_model with simple_save
# often used in training and serving stage

import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


def save_model(model_path):
    x_input = tf.placeholder(tf.float32, shape=[7, 7], name="x_input")
    x_reshape = tf.reshape(x_input, shape=[-1, 7, 7, 1], name='x_reshape')
    w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 5], stddev=0.1), name='w')
    conv = tf.nn.conv2d(x_reshape, w, [1, 1, 1, 1], padding="SAME", name='conv')

    w1 = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1), name='w1')
    w2 = tf.Variable(tf.truncated_normal(shape=[3, 3], stddev=0.1), name='w2')
    y = tf.matmul(w1, w2, name='y')

    with tf.Session() as sess:
        x_data = np.random.rand(7, 7)
        sess.run(tf.global_variables_initializer())
        sess.run([w, conv], feed_dict={x_input: x_data})
        tf.saved_model.simple_save(sess, model_path,
                                   inputs={'input0': x_input}, outputs={'output0': conv})
        # here we use simple_save method to save model,but actually tensorflow indeed use the
        # save_model_builder to save model behind with default signature KEY (as well as we defined as "my_signature")


def restore_model(model_path):
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)
        signature = meta_graph_def.signature_def

        # get tensor name
        in_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['input0'].name
        out_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['output0'].name

        # get tensor
        in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
        out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

        # run
        conv_r = sess.run(out_tensor, feed_dict={in_tensor: np.full((7, 7), 10)})
        print(conv_r)


if __name__ == '__main__':
    _model_path = './simple_save/'
    mode = 'mode_restore'  # 'mode_save' or 'mode_restore'
    if mode == 'mode_save':
        save_model(_model_path)
    if mode == 'mode_restore':
        restore_model(_model_path)
