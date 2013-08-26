cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# FIND_PACKAGE(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# FIND_PACKAGE(VISP REQUIRED)
# IF(VISP_FOUND)
# 	INCLUDE(${VISP_USE_FILE})
# ENDIF(VISP_FOUND)

# modules for OpenMP parallel computing
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)
#set(ROS_BUILD_TYPE Release)

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

rosbuild_add_library(mbtEdgeTracker src/mbtEdgeTracker.cpp)
rosbuild_add_library(kltFbTracker src/kltFbTracker.cpp)
rosbuild_add_library(cadModel src/cadModel.cpp)
rosbuild_add_library(superResolutionTracker src/superResolutionTracker.cpp)
rosbuild_add_library(GCoptimization include/gco/GCoptimization.cpp)
rosbuild_add_library(LinkedBlockList include/gco/LinkedBlockList.cpp)
rosbuild_add_executable(track_node src/track_node.cpp)
target_link_libraries(track_node mbtEdgeTracker)
target_link_libraries(track_node kltFbTracker)
target_link_libraries(track_node cadModel)
target_link_libraries(track_node superResolutionTracker)
target_link_libraries(track_node GCoptimization)
target_link_libraries(track_node LinkedBlockList)
# interesting if using visp in ROS
target_link_libraries(track_node libxml2.so)
# target_link_libraries(track_node  ${OpenCV_LIBS} )
# target_link_libraries(track_node  ${VISP_LIBS} )
