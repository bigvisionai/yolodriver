import os
import tempfile
from general_utils.common_utils import read_yaml, write_yaml, add_keys_for_value, download_file
from config import NONE_STR, ROOT, LOG_DIR_NAME, YOLOV7, TRAIN_DIR_NAME
from general_utils.data_utils import yolov7_write_yaml
from config import YOLO_DATA_KEYS

from general_utils.yolov7_main import parse_opt, main

YOLOV7_DIR = os.path.join(ROOT, YOLOV7)

WEIGHT_DIR = 'weights'


DOWNLOAD_URLS = {
    'YOLOv7tiny': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt',
    'YOLOv7x': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt',
    'YOLOv7': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt',
    'YOLOv7w6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt',
    'YOLOv7e6e': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt',
    'YOLOv7e6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt',
    'YOLOv7d6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt'
}

HYP_PARAMS = {
    'YOLOv7tiny': 'hyp.scratch.tiny.yaml',
    'YOLOv7x': 'hyp.scratch.p5.yaml',
    'YOLOv7': 'hyp.scratch.custom.yaml',
    'YOLOv7w6': 'hyp.scratch.p6.yaml',
    'YOLOv7e6e': 'hyp.scratch.p6.yaml',
    'YOLOv7e6': 'hyp.scratch.p6.yaml',
    'YOLOv7d6': 'hyp.scratch.p6.yaml'
}


def get_hyperparams_for_model(model_key):
    filename = HYP_PARAMS[model_key]
    return os.path.join(YOLOV7_DIR, 'data', filename)


def get_model_yaml_file(model_key):
    url = DOWNLOAD_URLS[model_key]
    filename = url.split('/')[-1].split('.')[0].split('_')[0] + '.yaml'
    filepath = os.path.join(YOLOV7_DIR, 'cfg', 'training', filename)
    return filepath


def update_model_yaml(model_yaml, num_classes):
    dic = read_yaml(model_yaml)
    dic[YOLO_DATA_KEYS.NUMBER_OF_CLASSES] = num_classes
    parent_data_dir = tempfile.TemporaryDirectory().name
    os.makedirs(parent_data_dir, exist_ok=True)
    filename = os.path.basename(model_yaml)
    save_path = os.path.join(parent_data_dir, filename)
    write_yaml(dic, save_path)
    return save_path


def supported_weights():
    supported_weights_dict = dict()
    supported_weights_str = []
    for name, url in DOWNLOAD_URLS.items():
        weight_pt = url.split('/')[-1]
        weight_ = weight_pt.split('.')[0]
        weight_str = f'{name} | {weight_pt} | {weight_}'
        supported_weights_str.append(weight_str)
        supported_weights_dict = add_keys_for_value(supported_weights_dict, [name, weight_pt, weight_], name)
    return supported_weights_dict, supported_weights_str


SUPPORTED_WEIGHTS, SUPPORTED_WEIGHTS_STR = supported_weights()


def get_num_classes_from_yaml(yaml_path):
    dic = read_yaml(yaml_path)
    return dic[YOLO_DATA_KEYS.NUMBER_OF_CLASSES]


def yolov7_train(args):
    supported_model_str = "\n".join(SUPPORTED_WEIGHTS_STR)
    assert args.weights in SUPPORTED_WEIGHTS, f'Weight: {args.weights} is not supported. ' \
                                              f'\nSupported weights:\n {supported_model_str}'

    data_yaml_path = yolov7_write_yaml(args.data_dir, args.data_yaml_filename)

    output_dir = args.output_dir
    if output_dir == NONE_STR:
        output_dir = os.path.join(ROOT, LOG_DIR_NAME, YOLOV7, TRAIN_DIR_NAME)

    opt = parse_opt(known=True)
    opt.data = data_yaml_path
    model_key = SUPPORTED_WEIGHTS[args.weights]
    download_url = DOWNLOAD_URLS[model_key]
    weight_pt_name = download_url.split('/')[-1]
    model_save_path = os.path.join(WEIGHT_DIR, weight_pt_name)
    download_file(download_url, model_save_path)
    opt.weights = model_save_path
    opt.img = args.image_size
    opt.epochs = args.epochs
    opt.batch_size = args.batch_size
    opt.project = output_dir
    opt.name = args.exp_name
    opt.device = args.device

    ############################
    model_yaml_path = get_model_yaml_file(model_key)
    num_classes = get_num_classes_from_yaml(data_yaml_path)
    updated_model_yaml_path = update_model_yaml(model_yaml_path, num_classes)
    opt.cfg = updated_model_yaml_path
    ###########################
    opt.hyp = get_hyperparams_for_model(model_key)

    main(opt)


if __name__ == '__main__':
    print(get_hyperparams_for_model('YOLOv7tiny'))
