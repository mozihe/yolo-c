//
// Created by zhujunheng on 24-3-30.
//

#include "yolov5_dnn.h"

void yolov5::YOLOv5Detector::initConfig(std::string onnxpath, int iw, int ih, std::vector<std::string> labels, float threshold_score, float threshold_nms) {
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold_score;
    this->threshold_nms = threshold_nms;
    this->labels = labels;
    this->net = cv::dnn::readNetFromONNX(onnxpath);
}

void yolov5::YOLOv5Detector::detect(cv::Mat &frame, std::vector<DetectResult> &results) {
    int w = frame.cols;
    int h = frame.rows;
    int max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));
    float x_factor = float(image.cols) / float(input_w);
    float y_factor = float(image.rows) / float(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    cv::Mat forward = this->net.forward();
    cv::Mat det_output(forward.size[1], forward.size[2], CV_32F, forward.ptr<float>());

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++) {
        float confidence = det_output.at<float>(i, 4);
        if (confidence < this->threshold_score) {
            continue;
        }
        std::cout << "confidence: " << confidence << std::endl;
        cv::Mat classes_scores = det_output.row(i).colRange(5, labels.size() + 5);
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
    cv::dnn::NMSBoxes(boxes, confidences, threshold_score, threshold_nms, indexes);

    for (int index : indexes) {
        DetectResult dr;
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.score = confidences[index];
        cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        cv::putText(frame, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        results.push_back(dr);
    }

    std::ostringstream ss;
    std::vector<double> layersTimings;
    double freq = cv::getTickFrequency() / 1000.0;
    double time = net.getPerfProfile(layersTimings) / freq;
    ss << "FPS: " << 1000 / time;
    putText(frame, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

void yolov5::YOLOv5Detector::run() {
    cv::VideoCapture capture(0);
    cv::Mat frame;
    std::vector<DetectResult> results;
    while (true) {
        capture.read(frame);
        detect(frame, results);
        cv::imshow("opencv", frame);
        char c = char(cv::waitKey(1));
        if (c == 27) {
            break;
        }
        results.clear();
    }
}