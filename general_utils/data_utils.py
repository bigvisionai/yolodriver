import os
from config import YOLO_DATA_KEYS
from general_utils.utils import read_yaml


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
