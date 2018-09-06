import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# class NodeLookup(object):
#     def __init__(self):
#         label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt';
#         uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
#
#         self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
#
#     def load(self, label_lookup_path, uid_lookup_path):
#         proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
#
#         uid_to_human = {}
#
#         for line in proto_as_ascii_lines:
#             line = line.strip('\n')  # remove \n
#             parsed_items = line.split('\t')  # split with \t
#             uid = parsed_items[0]
#             human_string = parsed_items[1]
#             uid_to_human[uid] = human_string
#
#         proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
#         node_id_to_uid = {}
#         for line in proto_as_ascii:
#             if line.startswith('  target_class:'):
#                 target_class = int(line.split(': ')[1])
#             if line.startswith('  target_class_string:'):
#                 target_class_string = line.split(': ')[1]
#                 node_id_to_uid[target_class] = target_class_string[1:-2]
#
#         node_id_to_name = {}
#
#         for key, value in node_id_to_uid.items():
#             name = uid_to_human[value]
#             node_id_to_name[key] = name
#
#         return node_id_to_name
#
#     def id_to_string(self, node_id):
#         if node_id not in self.node_lookup:
#             return 'no '
#         return self.node_lookup[node_id]

# load the model graph
with tf.gfile.FastGFile('../inception_model/tf_hub_inceptionV3_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # "image","bottleneck_txt")
    current_test_resource = "image"

    if current_test_resource == "bottleneck_txt":
        # load bottle neck txt file
        f = open('../images/test_images/bn_sunflowers/2.txt')
        bottle_neck_list = f.readline().split(',')
        data = np.zeros((1, 2048), dtype=float)
        data[0:] = bottle_neck_list[0:]

        predictions = sess.run(softmax_tensor, {'input/BottleneckInputPlaceholder:0': data})
        predictions = np.squeeze(predictions)  # transfer to one dimension

    elif current_test_resource == "image":
        # load image
        image_path = "../images/test_images/rose2.jpg"
        print("image path: ", image_path)
        # read image and decode
        image_data_raw = tf.gfile.FastGFile(image_path, 'rb').read()
        image_data = tf.image.decode_jpeg(image_data_raw)
        # covert to float [0,1]
        image_data_float = tf.image.convert_image_dtype(image_data, tf.float32)
        # resize image
        image_resize = tf.image.resize_images(image_data_float, (299, 299), method=tf.image.ResizeMethod.BILINEAR)
        # expand to shape of [N,W,H,C]
        image_resize_expand = tf.expand_dims(image_resize, 0)

        image_data_input = sess.run(image_resize_expand)  # The value of a feed cannot be a tf.Tensor object
        # print(image_data_input)

        print(image_data_input.dtype)

        predictions = sess.run(softmax_tensor, {'Placeholder:0': image_data_input})
        predictions = np.squeeze(predictions)  # transfer to one dimension

        # show image
        # img = Image.open(image_path)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

    # prediction sort
    top_k = predictions.argsort()[-5:][::-1]  # list[<start>:<stop>:<step>] -> [::-1]
    print(top_k)

    # class name (order from retrain output_labels.txt)
    class_name = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    for node_id in top_k:
        human_string = class_name[node_id]
        score = predictions[node_id]
        print('prediction: ', human_string, '  probability: ', score)
    print('')

    # show image
    # img = Image.open(image_path)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
