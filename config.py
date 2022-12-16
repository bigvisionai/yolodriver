import pathlib
import os
import tempfile
from general_utils.common_utils import download_file, unzip, get_git_zip_dir_name


class CodeDownload:
    def __init__(self, repository, git_username, git_tag=None, commit_sha=None):
        self.REPOSITORY = repository
        self. GITHUB_TAG = git_tag
        self.GITHUB_USERNAME = git_username
        self.COMMIT_SHA = commit_sha
        self.download()
        # # https://github.com/WongKinYiu/yolov7/archive/refs/heads/main.zip

    def download_tag_url(self):
        assert self.GITHUB_TAG is not None, "Tag is not provided. So can not create URL"
        return f"https://github.com/{self.GITHUB_USERNAME}/{self.REPOSITORY}/" \
               f"archive/refs/tags/{self.GITHUB_TAG}.zip"

    def download_commit_url(self):
        assert self.COMMIT_SHA is not None, "Commit SHA is not provided. So can not create URL"
        return f"https://github.com/{self.GITHUB_USERNAME}/{self.REPOSITORY}/" \
               f"archive/{self.COMMIT_SHA}.zip"

    def download_main_url(self):
        return f"https://github.com/{self.GITHUB_USERNAME}/{self.REPOSITORY}/archive/" \
               f"refs/heads/main.zip"

    def download(self):
        # https://github.com/ultralytics/yolov5/archive/refs/tags/v7.0.zip
        # https://github.com/meituan/YOLOv6/archive/refs/tags/0.2.1.zip
        rename_dir_path = os.path.join(ROOT, self.REPOSITORY)
        if os.path.exists(rename_dir_path):
            return
        if self.GITHUB_TAG:
            download_url = self.download_tag_url()

        elif self.COMMIT_SHA:
            download_url = self.download_commit_url()

        else:
            download_url = self.download_main_url()

        temp_dir = tempfile.TemporaryDirectory().name
        zip_file_name = f"{self.REPOSITORY}-{self.GITHUB_TAG}-{self.COMMIT_SHA}.zip"
        save_path = os.path.join(temp_dir, zip_file_name)
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
YOLOV7 = 'yolov7'

ROOT = pathlib.Path(__file__).parent.resolve()


# GitHub Username
GITHUB_USERNAME = {
    YOLOV5: 'ultralytics',
    YOLOV6: 'meituan',
    YOLOV7: 'WongKinYiu'
}

# There must be either version tag or commit sha for each repository

# version dict
GITHUB_TAG = {
    YOLOV5: 'v7.0',
    YOLOV6: '0.2.1'
}

# Commit SHA
GITHUB_COMMIT_SHA = {
    YOLOV7: '8c0bf3f78947a2e81a1d552903b4934777acfa5f'
}


SUPPORTED_MODEL_TYPE = [YOLOV5, YOLOV6, YOLOV7]

################################################################


def download_all_model_code():
    for model in SUPPORTED_MODEL_TYPE:
        CodeDownload(model, GITHUB_USERNAME[model], GITHUB_TAG.get(model, None),
                     GITHUB_COMMIT_SHA.get(model, None))


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
