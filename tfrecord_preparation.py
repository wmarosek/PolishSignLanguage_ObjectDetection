import os
import io
import tensorflow.compat.v1 as tf
import pandas as pd
import glob
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util, label_map_util

from PIL import Image
from collections import namedtuple

from params import IMAGE_DIR_OUT, WORKSPACE_PATH

# -------------------------------------------------------------------------------------
#   convert_xml_to_csv():
#   input: xml_dir
#   output: xml_df
# -------------------------------------------------------------------------------------
def convert_xml_to_csv(xml_dir):

    xml_list = []
    for xml_file in glob.glob(xml_dir + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


# -------------------------------------------------------------------------------------
#   class_text_to_int():
#   input: row_label, label_map_dict
#   output: label_map_dict[row_label]
# -------------------------------------------------------------------------------------
def class_text_to_int(row_label, label_map_dict):
    return label_map_dict[str(row_label)]


# -------------------------------------------------------------------------------------
#   create_tf_example():
#   input: group, path, label_map_dict
#   output: tf_example
# -------------------------------------------------------------------------------------
def create_tf_example(group, path, label_map_dict):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        [x.encode('utf-8') for x in label_map_dict]
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# -------------------------------------------------------------------------------------
#   split():
#   input: df, group
#   output: n/a
# -------------------------------------------------------------------------------------
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


# -------------------------------------------------------------------------------------
#   tfrecord_preparation():
#   input: out_directory_path, images_dir
#   output: n/a
# -------------------------------------------------------------------------------------
def tfrecord_preparation(out_directory_path, images_dir):

    label_path = os.path.join(out_directory_path, 'label_map.pbtxt')
    label_map = label_map_util.load_labelmap(label_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    tfrecord_out_dir = os.path.join(out_directory_path, 'tfrecords')
    if not os.path.exists(tfrecord_out_dir):
        os.makedirs(tfrecord_out_dir)

    images_train_dir = os.path.join(images_dir, 'train')
    images_test_dir = os.path.join(images_dir, 'test')

    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_out_dir, 'train.record'))
    examples = convert_xml_to_csv(images_train_dir)

    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_train_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the train TFRecord file: {}'.format(tfrecord_out_dir))

    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_out_dir, 'test.record'))
    examples = convert_xml_to_csv(images_test_dir)

    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_test_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the test TFRecord file: {}'.format(tfrecord_out_dir))



if __name__ == '__main__':
  tfrecord_preparation(out_directory_path=WORKSPACE_PATH, images_dir=IMAGE_DIR_OUT)