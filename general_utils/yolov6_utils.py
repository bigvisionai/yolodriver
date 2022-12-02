import os
import tempfile
import importlib.util
import sys


from general_utils.utils import write_yaml, download_file
from general_utils.data_utils import update_abs_path_in_yaml

from config import NONE_STR, ROOT, LOG_DIR_NAME, TRAIN_DIR_NAME, YOLOV6, YOLO_DATA_KEYS


YOLOV6_DIR = os.path.join(ROOT, YOLOV6)
MODEL_DEF_DIR = 'configs'

if str(YOLOV6_DIR) not in sys.path:
    sys.path.append(str(YOLOV6_DIR))

from YOLOv6.tools.train import get_args_parser, main


"""
Models Download Link for tag 0.2.1
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.pt
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.pt
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.pt
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l_relu.pt
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.pt

"""

MODEL_DOWNLOAD_TAG = '0.2.0'


def get_download_link(model_pt):
    return f"https://github.com/meituan/YOLOv6/releases/download/{MODEL_DOWNLOAD_TAG}/{model_pt}"


def add_keys_to_value(dic, keys, value):
    for key in keys:
        dic[key] = value
    return dic


def get_supported_model():
    model_dir = os.path.join(YOLOV6_DIR, MODEL_DEF_DIR)
    files = os.listdir(model_dir)
    models_str = []
    model_to_py = dict()
    model_download_link = dict()
    download_model_path = dict()
    for file in files:
        if file.endswith('finetune.py'):
            py_file_path = os.path.join(model_dir, file)
            spec = importlib.util.spec_from_file_location("module", py_file_path)
            finetune = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(finetune)
            model_dic = finetune.model
            model_name = model_dic['type']
            model_pt = model_dic['pretrained'].split('/')[-1]
            model_ = model_pt.split('.')[0]
            model_str = f'{model_name} | {model_pt}| {model_}'
            models_str.append(model_str)
            model_to_py = add_keys_to_value(model_to_py, [model_name, model_pt, model_], py_file_path)
            model_download_link = add_keys_to_value(model_download_link, [model_name, model_pt, model_],
                                                    get_download_link(model_pt))
            download_model_path = add_keys_to_value(download_model_path, [model_name, model_pt, model_],
                                                    model_dic['pretrained'])

    return models_str, model_to_py, model_download_link, download_model_path


SUPPORTED_MODELS_STR, MODEL_FINETUNE_PY, MODEL_DOWNLOAD_URL, MODEL_PATH = get_supported_model()


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


def yolov6_train(args):
    supported_model_str = "\n".join(SUPPORTED_MODELS_STR)
    assert args.weights in MODEL_FINETUNE_PY, f'Weight: {args.weights} is not supported. ' \
                                              f'\nSupported weights:\n {supported_model_str}'
    finetune_py_file = MODEL_FINETUNE_PY[args.weights]
    download_file(MODEL_DOWNLOAD_URL[args.weights], MODEL_PATH[args.weights])
    data_yaml_path = yolov6_write_yaml(args.data_dir, args.data_yaml_filename)

    output_dir = args.output_dir
    if output_dir == NONE_STR:
        output_dir = os.path.join(ROOT, LOG_DIR_NAME, YOLOV6, TRAIN_DIR_NAME)

    opt = get_args_parser().parse_args()
    opt.data_path = data_yaml_path
    opt.conf_file = finetune_py_file
    opt.image_size = args.image_size
    opt.eval_interval = args.eval_interval
    opt.epochs = args.epochs
    opt.batch_size = args.batch_size
    opt.device = args.device
    opt.output_dir = output_dir
    opt.name = args.exp_name
    ########################################
    opt.check_images = True
    opt.check_labels = True
    opt.write_trainbatch_tb = True

    main(opt)
    return


if __name__ == '__main__':
    # p = 'C:\Users\Prakash\pc\work\yolodriver\yolov5\v_data\train\images'
    path = 'C:\\Users\\Prakash\\pc\\work\\yolodriver\\yolov5\\v_data\\train\\images'
    data_dir = tempfile.TemporaryDirectory().name
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    print(yolo6_path_from_yolov5(path, images_dir, labels_dir))
