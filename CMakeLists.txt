cmake_minimum_required(VERSION 3.15)
project(v8_post)

set(CMAKE_CXX_STANDARD 11)

set(pybind11_DIR ${PYBIND11_PATH})
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(./yolov8_post)

set(SRC main.cpp yolov8_post/yolov8_post.cpp)

pybind11_add_module(v8_post MODULE ${SRC})
target_link_libraries(v8_post PRIVATE ${OpenCV_LIBS})

install(TARGETS v8_post DESTINATION ${CMAKE_SOURCE_DIR}/module)
