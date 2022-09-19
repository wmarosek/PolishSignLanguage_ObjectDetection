import os
import random
import re
from shutil import copyfile
import codecs
from PIL import Image
from params import IMAGE_DIR, IMAGE_DIR_OUT, WORKSPACE_PATH
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
            image_path = os.path.join(class_directory_path, image_filename)
            print(image_path)
            if os.path.exists(image_path):
                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpeg|.png)$', image_filename):
                    im1 = Image.open(os.path.join(class_directory_path, image_path))
                    width, height = im1.size
                    im1 = im1.convert('RGB')
                    im1 = im1.resize((width, height))
                    new_image_filename = image_filename.split('.')[0] + '.jpg'
                    im1.save(os.path.join(train_out_dir, new_image_filename))
                    print('Save as jpg:' + new_image_filename)

                    with open(os.path.join(class_directory_path, image_xml_filename), 'r') as file:
                        filedata = file.read()

                    filedata = filedata.replace('png', 'jpg')
                    filedata = filedata.replace('jpeg', 'jpg')

                    with open(os.path.join(train_out_dir, image_xml_filename), 'w') as file:
                        file.write(filedata)

                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg)$', image_filename):
                    copyfile(os.path.join(class_directory_path, image_filename), os.path.join(train_out_dir, image_filename))
                    copyfile(os.path.join(class_directory_path, image_xml_filename), os.path.join(train_out_dir, image_xml_filename))

            class_images.remove(image_filename)

        for image_filename in class_images:
            image_xml_filename = image_filename.split('.')[0] + '.xml'
            image_path = os.path.join(class_directory_path, image_filename)
            if os.path.exists(os.path.join(class_directory_path, image_xml_filename)):
                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpeg|.png)$', image_filename):
                    im1 = Image.open(os.path.join(class_directory_path, image_path))
                    width, height = im1.size
                    im1 = im1.convert('RGB')
                    im1 = im1.resize((width, height))
                    new_image_filename = image_filename.split('.')[0] + '.jpg'
                    im1.save(os.path.join(test_out_dir, new_image_filename))
                    print('Save as jpg:' + new_image_filename)

                    with open(os.path.join(class_directory_path, image_xml_filename), 'r') as file:
                        filedata = file.read()

                    filedata = filedata.replace('png', 'jpg')
                    filedata = filedata.replace('jpeg', 'jpg')

                    with open(os.path.join(test_out_dir, image_xml_filename), 'w') as file:
                        file.write(filedata)

                if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg)$', image_filename):
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


# -------------------------------------------------------------------------------------
#   prepare_label_map_file():  responsible for preparation label map file
#   input: _out_directory_path, _train_out_dir, _test_out_dir
#   output: n/a
# -------------------------------------------------------------------------------------
def prepare_label_map_file(_out_directory_path, _workspace_path, label_map_filename):
    if not os.path.exists(_out_directory_path):
        os.makedirs(_out_directory_path)

    if not os.path.exists(_workspace_path):
        os.makedirs(_workspace_path)

    label_map_filename = os.path.join(_workspace_path, label_map_filename)
    file = codecs.open(label_map_filename, "w", "utf-8")
    file_content = []

    class_list = os.listdir(_out_directory_path)
    for _id, class_detection in enumerate(class_list):
        class_id = _id + 1
        if _id == 0:
            file_content.append('item {\n')
        else:
            file_content.append('\n\nitem {\n')

        file_content.append(f'    id: {class_id}\n')
        file_content.append(f'    name: \"{class_detection}\"\n')
        file_content.append('}')

    file.write("".join(file_content))
    file.close()


if __name__ == '__main__':
    # divide_dataset(directory_path=IMAGE_DIR, out_directory_path=IMAGE_DIR_OUT, ratio=0.8)
    prepare_label_map_file(IMAGE_DIR, WORKSPACE_PATH, 'label_map.pbtxt')
