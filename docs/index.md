<div align="center">
<br>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/bigvisionai/yolodriver/actions/workflows/ci_actions.yml/badge.svg" alt="YoloDriver CI Actions"></a>
    <a href="https://hub.docker.com/r/opencvcourses/yolodriver"><img src="https://img.shields.io/docker/pulls/opencvcourses/yolodriver?logo=docker" alt="Docker Pulls"></a>
<br>
</div>

YoloDriver is one step to training and inference YOLOv5, YOLOv6, and Yolov7.

## Training

**Training of YOLOv5, YOLOv6, and YOLOv7 supported.**

### Training Help

```commandline
python train_driver.py -h
```

```commandline
usage: train_driver.py [-h] 
                       [--model_type MODEL_TYPE] 
                       [--weights WEIGHTS] 
                       [--data_dir DATA_DIR] 
                       [--data_yaml_filename DATA_YAML_FILENAME] 
                       [--image_size IMAGE_SIZE] [--epochs EPOCHS] 
                       [--batch_size BATCH_SIZE]
                       [--device DEVICE] 
                       [--output_dir OUTPUT_DIR] 
                       [--exp_name EXP_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        Which model type? Supported Models: 
                        ['yolov5', 'YOLOv6', 'yolov7']
  --weights WEIGHTS     Weight filename
  --data_dir DATA_DIR   Dataset directory
  --data_yaml_filename  DATA_YAML_FILENAME
                        Dataset YAML filename. Must be in data_dir
  --image_size IMAGE_SIZE
                        Image size (in pixels)
  --epochs EPOCHS       Max epochs to train
  --batch_size BATCH_SIZE
                        Batch size
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --output_dir OUTPUT_DIR
                        Path to save logs and trained-model. 
  --exp_name EXP_NAME   Name of the experiment.
```

### Train Data Format

**It supports the YOLO data format.**

YAML Data file example:

```commandline
train: train/images
val: valid/images

nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
```

Let us assume that the YAML filename above is `data.yaml`. 

The data YAML file should be in the data directory (`--data_dir`). 
Path in the YAML file should be with respect to the data directory.

### Training Examples

[Download the data sample](https://github.com/bigvisionai/yolodriver/releases/download/0.1/v_data.zip) and unzip it. 

**Let's assume the above-unzipped directory path is `v_data`.**

#### Finetune `yolov5n` from `YOLOv5`


```commandline
python train_driver.py --model_type yolov5
                       --weights yolov5n 
                       --data_dir v_data 
                       --data_yaml_filename data.yaml 
                       --image_size 640
                       --epochs 2
                       --batch_size 2
                       --device cpu
```

#### Finetune `yolov6t` from `YOLOv6`

```commandline
python train_driver.py --model_type YOLOv6
                       --weights yolov6t 
                       --data_dir v_data 
                       --data_yaml_filename data.yaml 
                       --image_size 640
                       --epochs 2
                       --batch_size 2
                       --device cpu
```


#### Finetune `yolov7tiny` from `YOLOv7`

```commandline
python train_driver.py --model_type yolov7
                       --weights YOLOv7tiny 
                       --data_dir v_data 
                       --data_yaml_filename data.yaml 
                       --image_size 640
                       --epochs 2
                       --batch_size 2
                       --device cpu
```

## Inference

**Inference of YOLOv5 supported.** 

### Inference Help

```commandline
python infer_driver.py -h 
```

```commandline
usage: infer_driver.py [-h]
                       [--model_type MODEL_TYPE]
                       [--weights WEIGHTS]
                       [--source SOURCE]
                       [--data_yaml_path DATA_YAML_PATH]
                       [--image_size IMAGE_SIZE [IMAGE_SIZE ...]]
                       [--conf_thres CONF_THRES]
                       [--iou_thres IOU_THRES]
                       [--device DEVICE]
                       [--view_img]
                       [--save_img_vid]
                       [--hide_labels]
                       [--hide_conf]
                       [--no_save_txt]
                       [--output_dir OUTPUT_DIR]
                       [--exp_name EXP_NAME]
                       [--exist_ok]
```

#### Inference using `yolov5n` from `YOLOv5`

```commandline
python infer_driver.py --model_type yolov5
                       --weights path/to/weight
                       --source image/dir/path or video/path or image/path
                       --data_yaml_path data/YAML/path
                       --image_size 640 640
                       --conf_thres 0.25
                       --iou_thres 0.45
                       --device cpu
                       --output_dir output/dir/path
                       --exp_name exp_name
```







