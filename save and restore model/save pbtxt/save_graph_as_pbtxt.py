import tensorflow as tf
import numpy as np


# save graph as pbtxt
def save_graph_as_pbtxt():
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

        graph_def = tf.get_default_graph().as_graph_def()
        tf.train.write_graph(graph_def, './pbtxt/','graph.pbtxt', as_text=True)


def convert_pb_to_pbtxt(pb_filename):
    with tf.gfile.GFile(pb_filename, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        # import_graph_def to import a serialized GraphDef and extract the tensor, op,
        # then place them to the default graph
        tf.import_graph_def(graph_def=restored_graph_def,
                                 #input_map=None,
                                 #return_elements=None,
                                 name=""  # the name position arg can't be ignore
                                 )
        
        graph_def = tf.get_default_graph().as_graph_def()
        tf.train.write_graph(graph_def, './pbtxt/','graph.pbtxt', as_text=True)


if __name__ == '__main__':
    # save_graph_as_pbtxt()
    convert_pb_to_pbtxt('frozen_model.pb')
