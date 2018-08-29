import tensorflow as tf
import os
import sys
import random
import math


def get_filenames_and_labels(dataset_dir):
    directories = []  # directories of image files
    label_names = []  # class label names
    for filename in os.listdir(dataset_dir):
        print(filename)
        path = os.path.join(dataset_dir, filename)
        print(path)
        if os.path.isdir(path):  # if a path
            directories.append(path)
            label_names.append(filename)

    image_filenames = []
    for dir in directories:
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            image_filenames.append(path)  # collect the paths of image files

    return image_filenames, label_names


def image_to_tfexample(image_data, image_format, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id]))
    }))


def get_output_filename(dataset_dir, split_name):
    output_filename = "tfrecords_%s.tfrecord/" % split_name
    return os.path.join(dataset_dir, output_filename)


def convert_dataset_to_tfrecords(split_name, filenames, class_name_to_ids, dataset_dir):
    assert split_name in ["train", "test"]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            output_filename = get_output_filename(dataset_dir, split_name)
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(len(filenames)):
                    try:
                        sys.stdout.write("\r>>converting %d iamges" % i)
                        sys.stdout.flush()
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()  # read image
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_name_to_ids[class_name]

                        example = image_to_tfexample(image_data, b'jpg', class_id)
                        tfrecord_writer.write(example.SerializeToString())
                    except IOError as e:
                        print("Could not read: ", filenames[i])
                        print("Error: ", e)
                        print("Skip it\n")

    sys.stdout.write('\n')
    sys.stdout.flush()


def write_label_file(label_to_class_names, dataset_dir, filename='LABELS_FILENAME'):
    lables_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(lables_filename, 'w') as f:
        for label in label_to_class_names:
            class_name = label_to_class_names[label]
            f.write("%d:%s\n" % (label, class_name))


if __name__ == "__main__":
    DATASET_DIR = "./tfRecords/images/"
    image_filenames, class_names = get_filenames_and_labels(DATASET_DIR)
    class_name_to_ids = dict(zip(class_names, range(len(class_names))))

    # split dataset to training and test data
    random.seed(0)
    random.shuffle(image_filenames)

    TRAIN_DATA_PERCENTAGE = 0.8
    NUM_TRAIN = int(len(image_filenames) * TRAIN_DATA_PERCENTAGE)
    training_filenames = image_filenames[:NUM_TRAIN]
    test_filenames = image_filenames[NUM_TRAIN:]

    # convert to tfRecords
    convert_dataset_to_tfrecords("train", training_filenames, class_name_to_ids, DATASET_DIR)
    convert_dataset_to_tfrecords("test", test_filenames, class_name_to_ids, DATASET_DIR)

    # output label file
    label_to_class_name = dict(zip(range(len(class_names)), class_names))
    write_label_file(label_to_class_name, DATASET_DIR)
