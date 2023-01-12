import argparse

from config import SUPPORTED_MODEL_TYPE, NONE_STR, ROOT, LOG_DIR_NAME, INFER_DIR_NAME, EXPERIMENT_NAME
from config import YOLOV5, YOLOV6, YOLOV7
from general_utils.common_utils import remove_add_dirs_to_sys_path


def argument_parser(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='yolov5',
                        help=f'Which model type? \nSupported Models: \n{SUPPORTED_MODEL_TYPE}')
    parser.add_argument('--weights', type=str, default='runs/yolov5/train/exp14/weights/best.pt', help='Weight filepath')
    parser.add_argument('--source', type=str, default='../v_data/valid/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data_yaml_path', type=str, default='../v_data/data.yaml',
                        help='To get class names')
    parser.add_argument('--image_size', '--img', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--view_img', default=False, action='store_true', help='show results')
    parser.add_argument('--save_img_vid', default=False, action='store_true',
                        help='Save image and video with prediction bounding box')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')

    parser.add_argument('--no_save_txt', default=False, action='store_true', help='do not save results to *.txt')

    parser.add_argument('--output_dir', type=str, default=NONE_STR,
                        help=f'Path to save logs and trained-model. Default is '
                             f'{ROOT}/{LOG_DIR_NAME}/model_type/{INFER_DIR_NAME}')
    parser.add_argument('--exp_name', type=str, default=EXPERIMENT_NAME, help='Name of the experiment.')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    assert opt.model_type in SUPPORTED_MODEL_TYPE, 'Model type: {} is not supported. Supported Models: {}'.\
        format(opt.model_type, SUPPORTED_MODEL_TYPE)

    # yolov5 training
    if opt.model_type == YOLOV5:
        remove_add_dirs_to_sys_path(SUPPORTED_MODEL_TYPE, [YOLOV5])
        from general_utils.yolov5_utils import yolov5_infer
        yolov5_infer(opt)

    # YOLOv6 training
    if opt.model_type == YOLOV6:
        assert False
        # remove_add_dirs_to_sys_path(SUPPORTED_MODEL_TYPE, [YOLOV6])
        # from general_utils.yolov6_utils import yolov6_train
        # yolov6_train(opt)

    # yolov7 training
    if opt.model_type == YOLOV7:
        assert False
        # remove_add_dirs_to_sys_path(SUPPORTED_MODEL_TYPE, [YOLOV7])
        # from general_utils.yolov7_utils import yolov7_train
        # yolov7_train(opt)


if __name__ == '__main__':
    args = argument_parser()
    main(args)
