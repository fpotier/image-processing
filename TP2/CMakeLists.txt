cmake_minimum_required(VERSION 3.14)
project(TP2)
add_definitions("-DPROJECT_DIRECTORY=\"${PROJECT_SOURCE_DIR}\"")

# OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenMP)

add_executable(circle-detector circle-detector.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(circle-detector PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(circle-detector PUBLIC ${OpenCV_LIBS})
endif()
