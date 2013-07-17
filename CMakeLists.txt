cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# FIND_PACKAGE(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# FIND_PACKAGE(VISP REQUIRED)
# IF(VISP_FOUND)
# 	INCLUDE(${VISP_USE_FILE})
# ENDIF(VISP_FOUND)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

rosbuild_add_library(track_ros src/track_ros.cpp)
rosbuild_add_library(KernelBasedTracking src/KernelBasedTracking.cpp)
rosbuild_add_library(CamShiftTracking src/CamShiftTracking.cpp)
rosbuild_add_library(medianFlowTracking src/medianFlowTracking.cpp)
rosbuild_add_library(houghLineBasedTracker src/houghLineBasedTracker.cpp)
rosbuild_add_library(mbtEdgeTracker src/mbtEdgeTracker.cpp)
rosbuild_add_library(kltFbTracker src/kltFbTracker.cpp)
rosbuild_add_library(cadModel src/cadModel.cpp)
rosbuild_add_library(activeModelTracker src/activeModelTracker.cpp)
rosbuild_add_library(nonparametricTextureLearningTracker src/nonparametricTextureLearningTracker.cpp)
rosbuild_add_library(textureTracker src/textureTracker.cpp)
# rosbuild_add_library(mbtKltTracker src/mbtKltTracker.cpp)
# rosbuild_add_library(mbtEdgeKltTracker src/mbtEdgeKltTracker.cpp)
rosbuild_add_executable(track_node src/track_node.cpp)
target_link_libraries(track_node track_ros)
target_link_libraries(track_node KernelBasedTracking)
target_link_libraries(track_node CamShiftTracking)
target_link_libraries(track_node medianFlowTracking)
target_link_libraries(track_node houghLineBasedTracker)
target_link_libraries(track_node mbtEdgeTracker)
target_link_libraries(track_node kltFbTracker)
target_link_libraries(track_node cadModel)
target_link_libraries(track_node activeModelTracker)
target_link_libraries(nonparametricTextureLearningTracker textureTracker)
target_link_libraries(track_node nonparametricTextureLearningTracker)
# target_link_libraries(track_node mbtKltTracker)
# target_link_libraries(track_node mbtEdgeKltTracker)
# target_link_libraries(track_node  ${OpenCV_LIBS} )
# target_link_libraries(track_node  ${VISP_LIBS} )
