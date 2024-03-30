//
// Created by zhujunheng on 24-3-30.
//

#include "yolov5_dnn.h"

using namespace yolov5;

void YOLOv5Detector::initConfig(std::string onnxpath, int iw, int ih, float threshold) {
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold;
    this->net = cv::dnn::readNetFromONNX(onnxpath);

    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    printf("唉，没cuda版本，只能用cpu了\n");
}

void YOLOv5Detector::detect(cv::Mat &frame, std::vector<DetectResult> &results) {
    int w = frame.cols;
    int h = frame.rows;
    int max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));
    float x_factor = float(image.cols) / float(input_w);
    float y_factor = float(image.rows) / float(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0),
                                          true, false);
    this->net.setInput(blob);
    cv::Mat preds = this->net.forward();
    cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++) {
        float confidence = det_output.at<float>(i, 4);
        if (confidence < this->threshold_score) {
            continue;
        }
        std::cout << "confidence: " << confidence << std::endl;
        cv::Mat classes_scores = det_output.row(i).colRange(5, 6);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &classIdPoint);

        float cx = det_output.at<float>(i, 0);
        float cy = det_output.at<float>(i, 1);
        float ow = det_output.at<float>(i, 2);
        float oh = det_output.at<float>(i, 3);
        int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
        int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
        int width = static_cast<int>(ow * x_factor);
        int height = static_cast<int>(oh * y_factor);
        cv::Rect box;
        box.x = x;
        box.y = y;
        box.width = width;
        box.height = height;

        boxes.push_back(box);
        classIds.push_back(classIdPoint.x);
        confidences.push_back(score);
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, threshold_score, 0.45, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        DetectResult dr;
        int index = indexes[i];
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.score = confidences[index];
        cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                      cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        results.push_back(dr);
    }

    std::ostringstream ss;
    std::vector<double> layersTimings;
    double freq = cv::getTickFrequency() / 1000.0;
    double time = net.getPerfProfile(layersTimings) / freq;
    ss << "FPS: " << 1000 / time << " ; time : " << time << " ms";
    putText(frame, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
