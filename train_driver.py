import argparse

from config import SUPPORTED_MODEL_TYPE, NONE_STR, ROOT, LOG_DIR_NAME, TRAIN_DIR_NAME, EXPERIMENT_NAME
from config import YOLOV5, YOLOV6, YOLOV7
from general_utils.common_utils import remove_add_dirs_to_sys_path


def argument_parser(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='yolov7',
                        help=f'Which model type? \nSupported Models: \n{SUPPORTED_MODEL_TYPE}')
    parser.add_argument('--weights', type=str, default='YOLOv7tiny', help='Weight filename')
    parser.add_argument('--data_dir', type=str, default='../v_data', help='Dataset directory')
    parser.add_argument('--data_yaml_filename', type=str, default='data.yaml',
                        help='Dataset YAML filename. Must be in data_dir')
    parser.add_argument('--image_size', type=int, default=640, help='Image size (in pixels)')
    parser.add_argument('--epochs', type=int, default=2, help='Max epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output_dir', type=str, default=NONE_STR,
                        help=f'Path to save logs and trained-model. Default is '
                             f'{ROOT}/{LOG_DIR_NAME}/model_type/{TRAIN_DIR_NAME}')
    parser.add_argument('--exp_name', type=str, default=EXPERIMENT_NAME, help='Name of the experiment.')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    assert opt.model_type in SUPPORTED_MODEL_TYPE, 'Model type: {} is not supported. Supported Models: {}'.\
        format(opt.model_type, SUPPORTED_MODEL_TYPE)

    # yolov5 training
    if opt.model_type == YOLOV5:
        remove_add_dirs_to_sys_path(SUPPORTED_MODEL_TYPE, [YOLOV5])
        from general_utils.yolov5_utils import yolov5_train
        yolov5_train(opt)

    # YOLOv6 training
    if opt.model_type == YOLOV6:
        remove_add_dirs_to_sys_path(SUPPORTED_MODEL_TYPE, [YOLOV6])
        from general_utils.yolov6_utils import yolov6_train
        yolov6_train(opt)

    # yolov7 training
    if opt.model_type == YOLOV7:
        remove_add_dirs_to_sys_path(SUPPORTED_MODEL_TYPE, [YOLOV7])
        from general_utils.yolov7_utils import yolov7_train
        yolov7_train(opt)


if __name__ == '__main__':
    args = argument_parser()
    main(args)





