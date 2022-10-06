cmake_minimum_required(VERSION 3.14)
project(TP1)
add_definitions("-DPROJECT_DIRECTORY=\"${PROJECT_SOURCE_DIR}\"")
set(CMAKE_CXX_FLAGS_RELEASE "-fdiagnostics-color=always -O3 -DNDEBUG -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-fdiagnostics-color=always -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(reconstruction reconstruction.cpp)
target_link_libraries(reconstruction ${OpenCV_LIBS})

add_executable(check check.cpp)
target_link_libraries(check ${OpenCV_LIBS})
