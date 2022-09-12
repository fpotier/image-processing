cmake_minimum_required(VERSION 3.14)
project(TP1)
add_definitions("-DPROJECT_DIRECTORY=\"${PROJECT_SOURCE_DIR}\"")
set(CMAKE_CXX_FLAGS_RELEASE " -O3 -DNDEBUG -Wall -Wextra")

# OpenCV
find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(reconstruction reconstruction.cpp)
target_link_libraries(reconstruction ${OpenCV_LIBS})
