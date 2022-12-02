import argparse

from config import SUPPORTED_MODEL_TYPE, YOLOV5, YOLOV6, NONE_STR, ROOT, LOG_DIR_NAME, TRAIN_DIR_NAME, EXPERIMENT_NAME
from general_utils.yolov5_utils import yolov5_train
from general_utils.yolov6_utils import yolov6_train


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='YOLOv6',
                        help=f'Which model type? \nSupported Models: \n{SUPPORTED_MODEL_TYPE}')
    parser.add_argument('--weights', type=str, default='yolov6n.pt', help='Weight filename')
    parser.add_argument('--data_dir', type=str, default='../yolodriver/yolov5/v_data', help='Dataset directory')
    parser.add_argument('--data_yaml_filename', type=str, default='data.yaml',
                        help='Dataset YAML filename. Must be in data_dir')
    parser.add_argument('--image_size', type=int, default=640, help='Image size (in pixels)')
    parser.add_argument('--epochs', type=int, default=2, help='Max epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output_dir', type=str, default=NONE_STR,
                        help=f'Path to save logs and trained-model. Default is '
                             f'{ROOT}/{LOG_DIR_NAME}/model_type/{TRAIN_DIR_NAME}')
    parser.add_argument('--exp_name', type=str, default=EXPERIMENT_NAME, help='Name of the experiment.')

    return parser.parse_args()


def main():
    opt = argument_parser()
    assert opt.model_type in SUPPORTED_MODEL_TYPE, 'Model type: {} is not supported. Supported Models: {}'.\
        format(opt.model_type, SUPPORTED_MODEL_TYPE)

    # yolov5 training
    if opt.model_type == YOLOV5:

        yolov5_train(opt)
    # YOLOv6 training
    if opt.model_type == YOLOV6:
        yolov6_train(opt)


if __name__ == '__main__':
    main()





