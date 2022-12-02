import pathlib
import os
from general_utils.utils import download_file, unzip, get_git_zip_dir_name


class CodeDownload:
    def __init__(self, repository, git_username, git_tag):
        self.REPOSITORY = repository
        self. GITHUB_TAG = git_tag
        self.GITHUB_USERNAME = git_username
        self.download()

    def download(self):
        # https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip
        # https://github.com/meituan/YOLOv6/archive/refs/tags/0.2.1.zip
        rename_dir_path = os.path.join(ROOT, self.REPOSITORY)
        if os.path.exists(rename_dir_path):
            return
        download_url = f"https://github.com/{self.GITHUB_USERNAME}/{self.REPOSITORY}/archive/refs/tags/" \
                       f"{self.GITHUB_TAG}.zip"
        zip_file_name_prefix = f"{self.REPOSITORY}-{self.GITHUB_TAG}"
        save_path = os.path.join(ROOT, f"{zip_file_name_prefix}.zip")
        download_file(download_url, save_path)
        unzip(save_path, ROOT)
        code_dir_name = get_git_zip_dir_name(save_path)
        code_dir_path = os.path.join(ROOT, code_dir_name)
        os.rename(code_dir_path, rename_dir_path)
        return


#################################################################
# Models
YOLOV5 = 'yolov5'
YOLOV6 = 'YOLOv6'

ROOT = pathlib.Path(__file__).parent.resolve()


# version dict
GITHUB_TAG = {
    YOLOV5: 'v7.0',
    YOLOV6: '0.2.1'
}


# GitHub Username
GITHUB_USERNAME = {
    YOLOV5: 'ultralytics',
    YOLOV6: 'meituan'
}


SUPPORTED_MODEL_TYPE = [YOLOV5, YOLOV6]
# SUPPORTED_MODEL_TYPE = [YOLOV6]

################################################################


def download_all_model_code():
    for model in SUPPORTED_MODEL_TYPE:
        CodeDownload(model, GITHUB_USERNAME[model], GITHUB_TAG[model])


download_all_model_code()

# General Universal Variable
NONE_STR = ''
LOG_DIR_NAME = 'runs'
TRAIN_DIR_NAME = 'train'
EXPERIMENT_NAME = 'exp'
CPU_STR = 'cpu'

# YOLO data format keys


class YoloDataAttributes:
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'val'
    NUMBER_OF_CLASSES = 'nc'
    CLASS_NAMES = 'names'
    IMAGES_DIR_NAME = 'images'
    LABELS_DIR_NAME = 'labels'


YOLO_DATA_KEYS = YoloDataAttributes()

