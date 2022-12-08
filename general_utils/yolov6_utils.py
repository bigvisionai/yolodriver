import os
import tempfile
import importlib.util
import sys


from general_utils.common_utils import write_yaml, download_file, add_keys_for_value
from general_utils.data_utils import update_abs_path_in_yaml, yolov6_write_yaml

from config import NONE_STR, ROOT, LOG_DIR_NAME, TRAIN_DIR_NAME, YOLOV6, YOLO_DATA_KEYS


YOLOV6_DIR = os.path.join(ROOT, YOLOV6)
MODEL_DEF_DIR = 'configs'


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
            model_to_py = add_keys_for_value(model_to_py, [model_name, model_pt, model_], py_file_path)
            model_download_link = add_keys_for_value(model_download_link, [model_name, model_pt, model_],
                                                     get_download_link(model_pt))
            download_model_path = add_keys_for_value(download_model_path, [model_name, model_pt, model_],
                                                     model_dic['pretrained'])

    return models_str, model_to_py, model_download_link, download_model_path


SUPPORTED_MODELS_STR, MODEL_FINETUNE_PY, MODEL_DOWNLOAD_URL, MODEL_PATH = get_supported_model()


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
