//
// Created by zhujunheng on 24-3-30.
//

#ifndef YOLOV5_DNN_H
#define YOLOV5_DNN_H


#include <opencv2/opencv.hpp>

namespace yolov5 {
    struct DetectResult {
        int classId;
        float score;
        cv::Rect box;
    };

    class YOLOv5Detector {
    public:
        void initConfig(std::string onnxpath, int iw, int ih, float threshold);
        void detect(cv::Mat & frame, std::vector<DetectResult> &result);
    private:
        int input_w = 640;
        int input_h = 480;
        cv::dnn::Net net;
        float threshold_score = 0.25;
    };
}

#endif //YOLOV5_DNN_H