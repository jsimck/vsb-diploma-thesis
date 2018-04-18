cmake_minimum_required(VERSION 3.6)
project(vsb-semestral-project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++14 -march=native -Wall -Wno-gnu-anonymous-struct -Wno-nested-anon-types -pedantic")

# Vizualization flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ_HASHING")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ_TPL_FEATURES")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ_MATCHING")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ_NMS")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ_OBJECTNESS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIZ_RESULTS")

# Source files
set(SOURCE_FILES
    main.cpp
    utils/visualizer.h utils/visualizer.cpp
    core/template.h core/template.cpp
    utils/parser.h utils/parser.cpp
    utils/timer.h utils/timer.cpp
    objdetect/hasher.h objdetect/hasher.cpp
    core/hash_key.h core/hash_key.cpp
    core/hash_table.h core/hash_table.cpp
    core/triplet.h core/triplet.cpp
    objdetect/classifier.h objdetect/classifier.cpp
    core/window.h core/window.cpp
    objdetect/matcher.h objdetect/matcher.cpp
    core/match.h core/match.cpp
    core/classifier_criteria.h core/classifier_criteria.cpp
    processing/processing.h processing/processing.cpp
    processing/computation.h
    core/camera.h core/camera.cpp
    core/scene.h core/scene.cpp
    utils/converter.h utils/converter.cpp
    utils/glutils.h utils/glutils.cpp
    glcore/mesh.h glcore/mesh.cpp
    glcore/shader.h glcore/shader.cpp
    libs/tinyply.h libs/tinyply.cpp
    core/particle.h core/particle.cpp
    glcore/frame_buffer.h glcore/frame_buffer.cpp
    objdetect/fine_pose.h objdetect/fine_pose.cpp
        utils/evaluator.h utils/evaluator.cpp core/result.cpp core/result.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
include_directories(${GLFW_INCLUDE_PATH})

find_package(glew REQUIRED)
include_directories(${GLEW_INCLUDE_PATH})

find_package(glfw3 REQUIRED)
include_directories(${GLM_INCLUDE_PATH})

find_package(glm REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

add_executable(vsb-semestral-project ${SOURCE_FILES})
target_link_libraries(vsb-semestral-project
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${OpenCV_LIBRARIES}
    glfw
    ${GLEW_LIBRARIES}
    ${GSL_LIBRARIES}
)
