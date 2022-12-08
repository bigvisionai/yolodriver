import os
from general_utils.common_utils import add_keys_for_value
from general_utils.data_utils import yolov5_write_yaml
from config import NONE_STR, ROOT, LOG_DIR_NAME, YOLOV5, TRAIN_DIR_NAME

from yolov5.train import parse_opt, main

WEIGHT_DIR = 'weights'


DOWNLOAD_URLS = {
    'YOLOv5n': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt',
    'YOLOv5s': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
    'YOLOv5m': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt',
    'YOLOv5l': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt',
    'YOLOv5x': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt',
    'YOLOv5n6': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n6.pt',
    'YOLOv5s6': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s6.pt',
    'YOLOv5m6': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m6.pt',
    'YOLOv5l6': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l6.pt',
    'YOLOv5x6': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt'
}


def supported_weights():
    supported_weights_dict = dict()
    supported_weights_str = []
    for name, url in DOWNLOAD_URLS.items():
        weight_pt = url.split('/')[-1]
        weight_ = weight_pt.split('.')[0]
        weight_str = f'{name} | {weight_pt} | {weight_}'
        supported_weights_str.append(weight_str)
        supported_weights_dict = add_keys_for_value(supported_weights_dict, [name, weight_pt, weight_], weight_pt)
    return supported_weights_dict, supported_weights_str


SUPPORTED_WEIGHTS, SUPPORTED_WEIGHTS_STR = supported_weights()


def yolov5_train(args):
    supported_weights_str = '\n'.join(SUPPORTED_WEIGHTS_STR)
    assert args.weights in SUPPORTED_WEIGHTS, f'Weight {args.weights} is not supported. ' \
                                              f'Supported weights:\n {supported_weights_str}'
    yaml_write_path = yolov5_write_yaml(args.data_dir, args.data_yaml_filename)

    output_dir = args.output_dir
    if output_dir == NONE_STR:
        output_dir = os.path.join(ROOT, LOG_DIR_NAME, YOLOV5, TRAIN_DIR_NAME)

    opt = parse_opt()
    opt.data = yaml_write_path
    opt.weights = os.path.join(WEIGHT_DIR, SUPPORTED_WEIGHTS[args.weights])
    opt.img = args.image_size
    opt.epochs = args.epochs
    opt.batch_size = args.batch_size
    opt.project = output_dir
    opt.name = args.exp_name

    main(opt)
    return




