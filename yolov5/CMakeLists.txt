cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

add_library(yolov5 yolov5_dnn.cpp yolov5_dnn.h)
target_link_libraries(yolov5 ${OpenCV_LIBS})

target_include_directories(yolov5 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})