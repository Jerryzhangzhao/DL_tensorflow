import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()

        uid_to_human = {}

        for line in proto_as_ascii_lines:
            line = line.strip('\n')  # remove \n
            parsed_items = line.split('\t')  # split with \t
            uid = parsed_items[0]
            human_string = parsed_items[1]
            uid_to_human[uid] = human_string

        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        node_id_to_name = {}

        for key, value in node_id_to_uid.items():
            name = uid_to_human[value]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return 'no '
        return self.node_lookup[node_id]


with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    for root, dirs, files in os.walk('images/'):
        for file in files:
            # load image
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)  # transfer to one dimension

            # print image path and name
            image_path = os.path.join(root, file)
            print("path:", image_path)

            # prediction sort
            top_k = predictions.argsort()[-5:][::-1]  # list[<start>:<stop>:<step>] -> [::-1]
            print(top_k)
            node_lookup = NodeLookup()

            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                print('prediction: ', human_string, '  probability: ', score)
            print('')

            # show image
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
