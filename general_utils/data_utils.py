import os
import tempfile
from config import YOLO_DATA_KEYS
from general_utils.common_utils import read_yaml, write_yaml


def is_data_path_relative_to_data_dir(data_dir, yaml_dic):
    train = yaml_dic.get(YOLO_DATA_KEYS.TRAIN, None)
    val = yaml_dic.get(YOLO_DATA_KEYS.VALID, None)
    test = yaml_dic.get(YOLO_DATA_KEYS.TEST, None)
    if train:
        abs_train_dir = os.path.join(data_dir, train)
        abs_train_dir = os.path.normpath(abs_train_dir)
        assert os.path.isdir(abs_train_dir), "YAML train path is not relative to data directory " \
                                             "or the path does not exist"
        yaml_dic[YOLO_DATA_KEYS.TRAIN] = abs_train_dir
    if val:
        abs_val_dir = os.path.join(data_dir, val)
        abs_val_dir = os.path.normpath(abs_val_dir)
        assert os.path.isdir(abs_val_dir), "YAML val path is not relative to data directory " \
                                           "or the path does not exist"
        yaml_dic[YOLO_DATA_KEYS.VALID] = abs_val_dir
    if test:
        abs_test_dir = os.path.join(data_dir, test)
        abs_test_dir = os.path.normpath(abs_test_dir)
        assert os.path.isdir(abs_test_dir), "YAML test path is not relative to data directory " \
                                            "or the path does not exist"
        yaml_dic[YOLO_DATA_KEYS.TEST] = abs_test_dir

    return yaml_dic


def update_abs_path_in_yaml(data_dir, yaml_filename):
    data_dir = os.path.abspath(data_dir)
    yaml_path = os.path.join(data_dir, yaml_filename)

    assert os.path.isfile(yaml_path), f"Data YAML file: {yaml_filename} does not exist in directory: {data_dir}"
    dic = read_yaml(yaml_path)
    dic = is_data_path_relative_to_data_dir(data_dir, dic)
    return dic


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


def yolov5_write_yaml(data_dir, yaml_filename):
    dic = update_abs_path_in_yaml(data_dir, yaml_filename)
    parent_data_dir = tempfile.TemporaryDirectory().name
    os.makedirs(parent_data_dir, exist_ok=True)
    data_yaml_save_path = os.path.join(parent_data_dir, yaml_filename)
    write_yaml(dic, data_yaml_save_path)
    return data_yaml_save_path


def yolov6_write_yaml(data_dir, yaml_filename):
    dic = update_abs_path_in_yaml(data_dir, yaml_filename)
    data_dir, dic = yolov6_create_symlink_update_dic(dic)
    data_yaml_path = os.path.join(data_dir, yaml_filename)
    write_yaml(dic, data_yaml_path)
    return data_yaml_path


# YOLO-v6 and YOLO-v7 has the same data format
yolov7_write_yaml = yolov6_write_yaml
