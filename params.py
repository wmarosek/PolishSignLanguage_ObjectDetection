import os
# -------------------------------------------------------------------------------------
#   Static file/directories paths
# -------------------------------------------------------------------------------------
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(CURRENT_PATH, 'database')
IMAGE_DIR_OUT = os.path.join(CURRENT_PATH, 'database_prepared')
XML_DIR = IMAGE_DIR_OUT
LABELS_PATH = ''
OUTPUT_PATH = ''
WORKSPACE_PATH = os.path.join(CURRENT_PATH, 'workspace')

# -------------------------------------------------------------------------------------
