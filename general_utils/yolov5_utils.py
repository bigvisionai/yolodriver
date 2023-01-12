import os
import argparse
from general_utils.common_utils import add_keys_for_value, print_args, is_model_pt_file_exist
from general_utils.data_utils import yolov5_write_yaml
from config import NONE_STR, ROOT, LOG_DIR_NAME, YOLOV5, TRAIN_DIR_NAME, INFER_DIR_NAME

from yolov5.train import parse_opt, main
from yolov5.detect import main as infer_main

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

    opt = parse_opt(known=True)
    opt.data = yaml_write_path
    opt.weights = os.path.join(WEIGHT_DIR, SUPPORTED_WEIGHTS[args.weights])
    opt.imgsz = args.image_size
    opt.epochs = args.epochs
    opt.batch_size = args.batch_size
    opt.project = output_dir
    opt.name = args.exp_name
    opt.device = args.device

    main(opt)
    return


def infer_parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return parser.parse_known_args()[0] if known else parser.parse_args()


def yolov5_infer(args):
    pt_exist = is_model_pt_file_exist(args.weights)
    if not pt_exist:
        supported_weights_str = '\n'.join(SUPPORTED_WEIGHTS_STR)
        weight_name = os.path.basename(args.weights)
        assert weight_name in SUPPORTED_WEIGHTS, f'Weight {args.weights} is not supported. ' \
                                                 f'Supported weights:\n {supported_weights_str}'
        dir_name = os.path.dirname(args.weights)
        if dir_name == '':
            dir_name = WEIGHT_DIR
        weights_path = os.path.join(dir_name, SUPPORTED_WEIGHTS[weight_name])
    else:
        weights_path = args.weights

    output_dir = args.output_dir
    if output_dir == NONE_STR:
        output_dir = os.path.join(ROOT, LOG_DIR_NAME, YOLOV5, INFER_DIR_NAME)

    opt = infer_parse_opt(known=True)

    opt.weights = weights_path

    opt.source = args.source
    opt.data = args.data_yaml_path

    opt.imgsz = args.image_size
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    opt.conf_thres = args.conf_thres
    opt.iou_thres = args.iou_thres

    opt.device = args.device

    opt.view_img = args.view_img
    opt.nosave = not args.save_img_vid
    opt.hide_labels = args.hide_labels
    opt.hide_conf = args.hide_conf

    opt.project = output_dir
    opt.name = args.exp_name
    opt.exist_ok = args.exist_ok

    opt.save_txt = not args.no_save_txt
    opt.save_conf = True

    print_args(vars(opt))

    infer_main(opt)
    return
