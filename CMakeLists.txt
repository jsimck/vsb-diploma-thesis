cmake_minimum_required(VERSION 3.6)
project(vsb-semestral-project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -Wall -pedantic")

set(SOURCE_FILES main.cpp core/template.cpp core/template.h objdetect/objectness.cpp objdetect/objectness.h utils/parser.cpp utils/parser.h utils/timer.h utils/utils.h objdetect/hasher.cpp objdetect/hasher.h core/hash_key.cpp core/hash_key.h core/hash_table.cpp core/hash_table.h core/triplet.cpp core/triplet.h objdetect/classifier.cpp objdetect/classifier.h core/window.cpp core/window.h utils/utils.cpp objdetect/matcher.cpp objdetect/matcher.h core/match.cpp core/match.h core/dataset_info.cpp core/dataset_info.h core/triplet_params.cpp core/triplet_params.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vsb-semestral-project ${SOURCE_FILES})
target_link_libraries(vsb-semestral-project ${OpenCV_LIBS})