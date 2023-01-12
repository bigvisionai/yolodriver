import os
import tempfile
import sys
import time

root = os.path.dirname(__file__)
root = os.path.dirname(root)

if root not in sys.path:
    sys.path.insert(0, root)

from train_driver import argument_parser, main
from infer_driver import main as infer_main
from infer_driver import argument_parser as infer_argument_parser
from general_utils.common_utils import unzip, download_file


def download_and_unzip_data():
    data_dir = tempfile.TemporaryDirectory().name
    os.makedirs(data_dir, exist_ok=True)
    data_url = 'https://github.com/bigvisionai/yolodriver/releases/download/0.1/v_data.zip'
    download_path = os.path.join(data_dir, 'v_data.zip')
    download_file(data_url, download_path)
    unzip(download_path, data_dir)
    return os.path.join(data_dir, 'v_data')


data_path = download_and_unzip_data()

YOLOv5_RUNS_DIR = 'yolov5_runs'
YOLOv5_EXP_NAME = 'exp_{}'.format(int(time.time()))


def test_yolov5_training():
    args = argument_parser(known=True)
    args.model_type = 'yolov5'
    args.weights = 'YOLOv5n'
    args.data_dir = data_path
    args.data_yaml_filename = 'data.yaml'
    args.image_size = 640
    args.epochs = 1
    args.batch_size = 2
    args.device = 'cpu'
    args.output_dir = YOLOv5_RUNS_DIR
    args.exp_name = YOLOv5_EXP_NAME
    main(args)


def test_yolov5_infer():
    args = infer_argument_parser(known=True)
    args.model_type = 'yolov5'
    args.weights = os.path.join(YOLOv5_RUNS_DIR, YOLOv5_EXP_NAME, 'weights', 'best.pt')
    args.source = os.path.join(data_path, 'valid', 'images')
    args.data_yaml_path = os.path.join(data_path, 'data.yaml')
    args.image_size = [640]
    args.conf_thres = 0.25
    args.batch_size = 0.45
    args.device = 'cpu'
    infer_main(args)


def test_yolov6_training():
    args = argument_parser(known=True)
    args.model_type = 'YOLOv6'
    args.weights = 'yolov6t'
    args.data_dir = data_path
    args.data_yaml_filename = 'data.yaml'
    args.image_size = 640
    args.epochs = 1
    args.batch_size = 2
    args.device = 'cpu'
    main(args)


def test_yolov7_training():
    args = argument_parser(known=True)
    args.model_type = 'yolov7'
    args.weights = 'YOLOv7tiny'
    args.data_dir = data_path
    args.data_yaml_filename = 'data.yaml'
    args.image_size = 640
    args.epochs = 1
    args.batch_size = 2
    args.device = 'cpu'
    main(args)
