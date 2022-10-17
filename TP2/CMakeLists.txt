cmake_minimum_required(VERSION 3.14)
project(TP2)
add_definitions("-DPROJECT_DIRECTORY=\"${PROJECT_SOURCE_DIR}\"")

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(circle-detector circle-detector.cpp)
target_link_libraries(circle-detector ${OpenCV_LIBS})
