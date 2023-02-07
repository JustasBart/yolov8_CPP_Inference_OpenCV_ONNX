# yolov8/yolov5 Inference C++

yolov8s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

yolov5s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)


This repository is based on OpenCVs dnn API to run an ONNX exported model of either yolov5/yolov8 (In theory should work for yolov6 and yolov7 but not tested). Note that for this example the networks are exported as rectangular (640x480) resolutions, but it would work for any resolution that you export as although you might want to use the letterBox approach for square images depending on your use-case.

It's based on using Qt as a GUI wrapper but it's built with CMake so it should be easy to strip down bits in order to suit individual projects.
Note that this repository is based on Ubuntu Linux, but that it should be trivial to simply re-use the Inference class or just the transpose piece of it to make it work in any environment.
