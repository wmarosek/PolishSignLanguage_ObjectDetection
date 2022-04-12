import os
import random
import re
from shutil import copyfile


# -------------------------------------------------------------------------------------
#   divide_dataset():  responsible for dividing dataset into testing and training sets
#   input: directory_path, out_directory_path, ratio
#   output: n/a
# -------------------------------------------------------------------------------------
def divide_dataset(directory_path, out_directory_path, ratio):
    train_out_dir = os.path.join(out_directory_path, 'train')
    test_out_dir = os.path.join(out_directory_path, 'test')
    prepare_directory_structure(out_directory_path, train_out_dir, test_out_dir)

    class_list = os.listdir(directory_path)

    for class_detection in class_list:
        class_directory_path = os.path.join(directory_path, class_detection)
        class_images = [
            filename for filename in os.listdir(class_directory_path) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', filename)
        ]

        for counter in range(int(len(class_images)*ratio)):
            image_filename = class_images[random.randint(0, len(class_images) - 1)]
            image_xml_filename = image_filename.split('.')[0] + '.xml'
            if os.path.exists(os.path.join(class_directory_path, image_xml_filename)):
                copyfile(os.path.join(class_directory_path, image_filename), os.path.join(train_out_dir, image_filename))
                copyfile(os.path.join(class_directory_path, image_xml_filename), os.path.join(train_out_dir, image_xml_filename))

            class_images.remove(image_filename)

        for image_filename in class_images:
            image_xml_filename = image_filename.split('.')[0] + '.xml'
            if os.path.exists(os.path.join(class_directory_path, image_xml_filename)):
                copyfile(os.path.join(class_directory_path, image_filename), os.path.join(test_out_dir, image_filename))
                copyfile(os.path.join(class_directory_path, image_xml_filename), os.path.join(test_out_dir, image_xml_filename))


# -------------------------------------------------------------------------------------
#   divide_dataset():  responsible for preparation of correct dataset project strucute
#   input: _out_directory_path, _train_out_dir, _test_out_dir
#   output: n/a
# -------------------------------------------------------------------------------------
def prepare_directory_structure(_out_directory_path, _train_out_dir, _test_out_dir):
    if not os.path.exists(_out_directory_path):
        os.makedirs(_out_directory_path)

    if not os.path.exists(_train_out_dir):
        os.makedirs(_train_out_dir)

    if not os.path.exists(_test_out_dir):
        os.makedirs(_test_out_dir)

