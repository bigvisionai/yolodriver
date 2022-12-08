import sys

import yaml
import os
import requests
import zipfile


def download_file(url, save_path):
    if not os.path.exists(save_path):
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)
        file = requests.get(url)
        open(save_path, 'wb').write(file.content)
    return


def unzip(zip_file_path, unzip_dir_path):
    print(zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(unzip_dir_path)


def get_git_zip_dir_name(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        first_file = zip_file.namelist()[0]
    dir_name = os.path.split(first_file)[0]
    return dir_name


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        try:
            dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return dic


def write_yaml(dic, yaml_path):
    with open(yaml_path, 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)
    return


def add_keys_for_value(dic, keys, value):
    for key in keys:
        dic[key] = value
    return dic


def remove_add_dirs_to_sys_path(remove, add):
    root = os.path.dirname(__file__)
    root = os.path.dirname(root)
    for d in remove:
        abs_path = os.path.join(root, d)
        print(abs_path)
        if abs_path in sys.path:
            sys.path.remove(abs_path)
    for d in add:
        abs_path = os.path.join(root, d)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
    return


def yolo6_path_from_yolov5(images_path, v6_images_path, v6_labels_path):
    images_path = os.path.normpath(images_path)
    images_path_splits = images_path.split(os.sep)
    labels_path_splits = images_path_splits.copy()
    labels_path_splits[-1] = YOLO_DATA_KEYS.LABELS_DIR_NAME
    labels_path = os.sep.join(labels_path_splits)

    v6_images_path = os.path.join(v6_images_path, images_path_splits[-2])
    v6_labels_path = os.path.join(v6_labels_path, images_path_splits[-2])

    # create symlink
    os.symlink(images_path, v6_images_path, target_is_directory=True)
    os.symlink(labels_path, v6_labels_path, target_is_directory=True)

    return v6_images_path


def yolov6_create_symlink_update_dic(yaml_dic):
    train = yaml_dic.get(YOLO_DATA_KEYS.TRAIN, None)
    val = yaml_dic.get(YOLO_DATA_KEYS.VALID, None)
    test = yaml_dic.get(YOLO_DATA_KEYS.TEST, None)

    parent_data_dir = tempfile.TemporaryDirectory().name
    v6_images_path = os.path.join(parent_data_dir, YOLO_DATA_KEYS.IMAGES_DIR_NAME)
    v6_labels_path = os.path.join(parent_data_dir, YOLO_DATA_KEYS.LABELS_DIR_NAME)
    os.makedirs(v6_images_path, exist_ok=True)
    os.makedirs(v6_labels_path, exist_ok=True)

    if train:
        yaml_dic[YOLO_DATA_KEYS.TRAIN] = yolo6_path_from_yolov5(train, v6_images_path, v6_labels_path)
    if val:
        yaml_dic[YOLO_DATA_KEYS.VALID] = yolo6_path_from_yolov5(val, v6_images_path, v6_labels_path)
    if test:
        yaml_dic[YOLO_DATA_KEYS.TEST] = yolo6_path_from_yolov5(test, v6_images_path, v6_labels_path)

    return parent_data_dir, yaml_dic


def yolov6_write_yaml(data_dir, yaml_filename):
    dic = update_abs_path_in_yaml(data_dir, yaml_filename)
    data_dir, dic = yolov6_create_symlink_update_dic(dic)
    data_yaml_path = os.path.join(data_dir, yaml_filename)
    write_yaml(dic, data_yaml_path)
    return data_yaml_path





