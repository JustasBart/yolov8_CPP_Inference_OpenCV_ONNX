# yolov8/yolov5 Inference C++

yolov8s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217226932-857b17f6-6868-411e-b90d-86b68075dc32.png)

yolov5s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217227085-15309527-88ea-4d4a-b737-fb1227811640.png)


This repository is based on OpenCVs dnn API to run an ONNX exported model of either yolov5/yolov8 (In theory should work for yolov6 and yolov7 but not tested). Note that for this example the networks are exported as rectangular (640x480) resolutions, but it would work for any resolution that you export as although you might want to use the letterBox approach for square images depending on your use-case.

It's based on using Qt as a GUI wrapper but it's built with CMake so it should be easy to strip down bits in order to suit individual projects.
Note that this repository is based on Ubuntu Linux, but that it should be trivial to simply re-use the Inference class or just the transpose piece of it to make it work in any environment.

The main interest here is the inference.cpp file which shows how to transpose yolov8 models to work as yolov5 models.

This code is subject to change in the future whenever there's a better (Cleaner/more efficient) way of achieving this transpose operation.
