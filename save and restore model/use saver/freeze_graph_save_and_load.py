# in this case, we save the frozen model to one single .pb model file
# and load it for forward inference

# for save model, first import meta graph and restore variables for checkpoint file
#     and then send the GraphDef to graph_util.convert_variables_to_constants and get
#     a new GraphDef object which is a simplified version of original one,
#     and then serialize the GraphDef and write it to disk using tf.gfile
#
# for load model, first we read the .pb file from disk and deserialize it,as a return we
#     get a GraphDef object, and then import it into the default graph use import_graph_def,
#     then we can get tensor from the graph for inference.
#

import tensorflow as tf
import numpy as np


def freeze_and_save_model():
    # import meta graph
    saver = tf.train.import_meta_graph('./saved_model_ckpt/model.ckpt.meta',clear_devices=True)
    graph = tf.get_default_graph()

    # get a GraphDef Object
    input_graph_def = graph.as_graph_def()

    # restore variables
    sess = tf.Session()
    saver.restore(sess, './saved_model_ckpt/model.ckpt')

    # convert_variables_to_constants receive a GraphDef object and session and
    # return a simplified version GraphDef of the original
    output_node_name = 'x_input,conv'
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                    input_graph_def=input_graph_def,
                                                                    output_node_name=output_node_name.split(','))
    # serialize to graph and write to disk
    output_graph_def_filename = './frozen_model.pb'
    with tf.gfile.GFile(output_graph_def_filename, 'wb')as f:
        f.write(output_graph_def.SerializeToString())

    sess.close()


def load_frozen_model():
    frozen_model_name = './frozen_model.pb'
    with tf.gfile.GFile(frozen_model_name, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        # import_graph_def to import a serialized GraphDef and extract the tensor, op,
        # then place them to the default graph
        aa = tf.import_graph_def(graph_def=restored_graph_def,
                                 #input_map=None,
                                 #return_elements=None,
                                 name=""  # the name position arg can't be ignore
                                 )
        print(aa)

    input_tensor = graph.get_tensor_by_name('x_input:0')
    conv_tensor = graph.get_tensor_by_name('conv:0')

    sess = tf.Session(graph=graph)
    conv_r = sess.run(conv_tensor, feed_dict={input_tensor: np.full((7, 7), 10)})
    print(conv_r)

    sess.close()


if __name__ == '__main__':
    mode = 'load'  # 'freeze' or 'load'
    if mode == 'freeze':
        freeze_and_save_model()
    if mode == 'load':
        load_frozen_model()
