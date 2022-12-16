import os
import tempfile

from train_driver import argument_parser, main
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
    main(args)


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





