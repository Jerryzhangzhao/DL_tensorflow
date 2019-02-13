# save and restore model using tf.saved_model with saved_model_builder
# often used in training and serving stage

import tensorflow as tf
import numpy as np


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
        _ = sess.run(conv, feed_dict={x_input: x_data})

        # construct saved model builder
        builder = tf.saved_model.builder.SavedModelBuilder(model_path)

        # build inputs and outputs dict,enable us to customize the inputs and outputs tensor name
        # when using the model, we don't need to care the tensor name define in the original graph
        inputs = {'input0': tf.saved_model.utils.build_tensor_info(x_input)}
        outputs = {'output0': tf.saved_model.utils.build_tensor_info(conv)}
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME

        # builder a signature
        my_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, method_name)

        # add meta graph and variables
        builder.add_meta_graph_and_variables(sess, ['MODEL_TRAINING'], signature_def_map={'my_signature': my_signature})

        # add_meta_graph method need add_meta_graph_and_variables method been invoked before
        builder.add_meta_graph(['MODEL_SERVING'], signature_def_map={'my_signature': my_signature})

        # save the model
        builder.save()


def restore_model(model_path):
    with tf.Session() as sess:
        # load model
        meta_graph_def = tf.saved_model.loader.load(sess, ['MODEL_TRAINING'], model_path)

        # get signature
        signature = meta_graph_def.signature_def

        # get tensor name
        in_tensor_name = signature['my_signature'].inputs['input0'].name
        out_tensor_name = signature['my_signature'].outputs['output0'].name

        # get tensor
        in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
        out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

        # run
        conv_r = sess.run(out_tensor, feed_dict={in_tensor: np.full((7, 7), 10)})
        print(conv_r)


if __name__ == '__main__':
    _model_path = './saved_model_builder/'
    mode = 'mode_save'  # 'mode_save' or 'mode_restore'
    if mode == 'mode_save':
        save_model(_model_path)
    if mode == 'mode_restore':
        restore_model(_model_path)
