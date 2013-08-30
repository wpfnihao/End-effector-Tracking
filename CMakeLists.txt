cmake_minimum_required(VERSION 2.4.6)
PROJECT(endeffector_tracking)

#SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# VISP
FIND_PACKAGE(VISP REQUIRED)
IF(VISP_FOUND)
	INCLUDE(${VISP_USE_FILE})
ENDIF(VISP_FOUND)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(include)

#add_library(mbtEdgeTracker src/mbtEdgeTracker.cpp)
#add_library(kltFbTracker src/kltFbTracker.cpp)
#add_library(cadModel src/cadModel.cpp)
add_library(superResolutionTracker src/superResolutionTracker.cpp)
add_library(GCoptimization include/gco/GCoptimization.cpp)
add_library(LinkedBlockList include/gco/LinkedBlockList.cpp)
add_executable(track_node src/track_node.cpp)
#target_link_libraries(track_node mbtEdgeTracker)
#target_link_libraries(track_node kltFbTracker)
#target_link_libraries(track_node cadModel)
target_link_libraries(track_node superResolutionTracker)
target_link_libraries(track_node GCoptimization)
target_link_libraries(track_node LinkedBlockList)
# interesting if using visp in ROS
# target_link_libraries(track_node libxml2.so)

target_link_libraries(track_node  ${OpenCV_LIBS} )
# target_link_libraries(track_node  ${VISP_LIBS} )
