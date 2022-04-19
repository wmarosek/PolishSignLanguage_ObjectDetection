# -------------------------------------------------------------------------------------
#   main():  responsible for loading information and running corresponding script
#   input: n/a
#   output: n/a
# -------------------------------------------------------------------------------------
import dataset_preparation
from params import IMAGE_DIR, CURRENT_PATH, IMAGE_DIR_OUT, WORKSPACE_PATH
from tfrecord_preparation import tfrecord_preparation


def main():
    dataset_preparation.divide_dataset(directory_path=IMAGE_DIR, out_directory_path=IMAGE_DIR_OUT, ratio=0.8)
    tfrecord_preparation(out_directory_path=WORKSPACE_PATH, images_dir=IMAGE_DIR_OUT)

main()
