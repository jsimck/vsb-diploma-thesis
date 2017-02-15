cmake_minimum_required(VERSION 3.6)
project(vsb-semestral-project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_COMPILER "/usr/local/bin/gcc-6")
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-6")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp objdetect/matching.cpp objdetect/matching.h core/template.cpp core/template.h objdetect/objectness.cpp objdetect/objectness.h utils/template_parser.cpp utils/template_parser.h utils/timer.h utils/utils.cpp utils/utils.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vsb-semestral-project ${SOURCE_FILES})
target_link_libraries(vsb-semestral-project ${OpenCV_LIBS})