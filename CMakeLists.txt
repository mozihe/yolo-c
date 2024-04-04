cmake_minimum_required(VERSION 3.10)
project(yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_subdirectory(yolov5)

add_executable(yolo main.cpp yolov5/yolov5_dnn.cpp yolov5/yolov5_dnn.h)

target_link_libraries(yolo yolov5)
