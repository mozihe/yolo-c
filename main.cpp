#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>

#include "yolov5/yolov5_dnn.h"

using namespace yolov5;

std::vector<std::string> classNames = {
    "sunhance"
};

int main() {
    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
    detector->initConfig("/home/zhujunheng/best.onnx", 640, 640, classNames, 0.25f);
    detector->run();
}