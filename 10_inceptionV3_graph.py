import tensorflow as tf
import os
import tarfile
import requests

# # url for model download
inception_pretrained_model_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# mkdir for model save
inception_pretrained_model_dir = "inception_model"
if not os.path.exists(inception_pretrained_model_dir):
    os.mkdir(inception_pretrained_model_dir)

# get filename and filepath
filename = inception_pretrained_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrained_model_dir, filename)

# download the model
if not os.path.exists(filepath):
    print("download ", filename)
    r = requests.get(inception_pretrained_model_url, stream=True)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                print(".")
print("finish download: ", filename)

# extract file
tarfile.open(filepath, "r:gz").extractall(inception_pretrained_model_dir)

# model graph save file
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# # pretrained model file: classify_image_graph_def.pb
inception_graph_def_file = os.path.join(inception_pretrained_model_dir,"classify_image_graph_def.pb")


filepath = os.path.join(inception_pretrained_model_dir,"inception-v3")

with tf.Session() as sess:
    # create a graph to save the model graph
    with tf.gfile.FastGFile(inception_graph_def_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name="")

    # save the structure of graph
    write = tf.summary.FileWriter(log_dir,sess.graph)
    write.close()


