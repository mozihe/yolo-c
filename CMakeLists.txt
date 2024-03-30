cmake_minimum_required(VERSION 3.28)
project(yolo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(yolo main.cpp
        yolov5_dnn.cpp
        yolov5_dnn.h)

target_link_libraries(yolo ${OpenCV_LIBS})
