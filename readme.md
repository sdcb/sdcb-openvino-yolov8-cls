# OpenVINO.NET demo for yolov8 classification model

This project is a minimal demo for my [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET) project to infer yolov8 classification model.

## NuGet packages requirements
* Sdcb.OpenVINO
* Sdcb.OpenVINO.runtime.win-x64
* OpenCvSharp4
* OpenCvSharp4.runtime.win

## Brief Introduction to Model Inference
The yolov8n-cls model has 1000 classifications (the specific 1000 classifications can be found [here](https://github.com/ultralytics/ultralytics/blob/12e3eef844b7b5e298647c5d9bf7e1cc41dcf8e0/ultralytics/cfg/datasets/ImageNet.yaml#L18)).

This model has an input size of `1x3x224x224xF32` and an output size of `1x1000xF32`.

The code will read the [hen.jpg](./hen.jpg) and try infer the most probabely classification, in this case, the output as follows: 
```
class id=hen, score=0.59
preprocess time: 0.00ms
infer time: 1.65ms
postprocess time: 0.49ms
Total time: 2.14ms
```

## Steps to convert from PyTorch yolov8 model into OpenVINO

* Downloaded from [ultralytics official website](https://docs.ultralytics.com/models/yolov8/#supported-modes), specifically, it's `YOLOv8n-cls.pt`(5.27MB).
* Install python, and install `ultralytics`: `pip install ultralytics`
* Convert `YOLOv8n-cls.pt` into OpenVINO xml model via command: `yolo export model=yolov8n-cls.pt format=openvino`
* After convert, you will get `yolov8n-cls.xml`(116KB) and `yolov8n-cls.bin`(10.3MB) in *yolov8n-cls_openvino_model* folder.
