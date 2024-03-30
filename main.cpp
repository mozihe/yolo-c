#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>

#include "yolov5_dnn.h"

using namespace yolov5;

std::vector<std::string> classNames = {
    "sunhance"
};

int main() {
    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
    detector->initConfig("/home/zhujunheng/best.onnx", 640, 640, 0.25f);
    cv::VideoCapture capture(0);
    cv::Mat frame;
    std::vector<DetectResult> results;
    while (true) {
        capture.read(frame);
        detector->detect(frame, results);
        for (const DetectResult& dr : results) {
            cv::Rect box = dr.box;
            cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
        }
        cv::imshow("opencv", frame);
        char c = char(cv::waitKey(1));
        if (c == 27) { // ESC 退出
            break;
        }
        // reset for next frame
        results.clear();
    }
}