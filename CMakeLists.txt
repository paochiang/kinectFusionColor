cmake_minimum_required(VERSION 2.8)  
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
project(kinectFusionC)

#OPENCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

include_directories("C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/inc")
link_directories("C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/Lib/x64")

add_executable(kinectFusionC main.cpp)

TARGET_LINK_LIBRARIES(kinectFusionC ${OpenCV_LIBS} Kinect20.Fusion.lib Kinect20.lib)