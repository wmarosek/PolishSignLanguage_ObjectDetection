# -------------------------------------------------------------------------------------
#   main():  responsible for loading information and running corresponding script
#   input: n/a
#   output: n/a
# -------------------------------------------------------------------------------------
import dataset_preparation
from params import IMAGE_DIR, CURRENT_PATH, IMAGE_DIR_OUT


def main():
    dataset_preparation.divide_dataset(directory_path=IMAGE_DIR, out_directory_path=IMAGE_DIR_OUT, ratio=0.8)


main()
