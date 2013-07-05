# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pengfei/fuerte_workspace/endeffector_tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pengfei/fuerte_workspace/endeffector_tracking

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/pengfei/fuerte_workspace/endeffector_tracking/CMakeFiles /home/pengfei/fuerte_workspace/endeffector_tracking/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/pengfei/fuerte_workspace/endeffector_tracking/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named CamShiftTracking

# Build rule for target.
CamShiftTracking: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 CamShiftTracking
.PHONY : CamShiftTracking

# fast build rule for target.
CamShiftTracking/fast:
	$(MAKE) -f CMakeFiles/CamShiftTracking.dir/build.make CMakeFiles/CamShiftTracking.dir/build
.PHONY : CamShiftTracking/fast

#=============================================================================
# Target rules for targets named KernelBasedTracking

# Build rule for target.
KernelBasedTracking: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 KernelBasedTracking
.PHONY : KernelBasedTracking

# fast build rule for target.
KernelBasedTracking/fast:
	$(MAKE) -f CMakeFiles/KernelBasedTracking.dir/build.make CMakeFiles/KernelBasedTracking.dir/build
.PHONY : KernelBasedTracking/fast

#=============================================================================
# Target rules for targets named ROSBUILD_genmsg_cpp

# Build rule for target.
ROSBUILD_genmsg_cpp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_genmsg_cpp
.PHONY : ROSBUILD_genmsg_cpp

# fast build rule for target.
ROSBUILD_genmsg_cpp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_genmsg_cpp.dir/build.make CMakeFiles/ROSBUILD_genmsg_cpp.dir/build
.PHONY : ROSBUILD_genmsg_cpp/fast

#=============================================================================
# Target rules for targets named ROSBUILD_genmsg_lisp

# Build rule for target.
ROSBUILD_genmsg_lisp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_genmsg_lisp
.PHONY : ROSBUILD_genmsg_lisp

# fast build rule for target.
ROSBUILD_genmsg_lisp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_genmsg_lisp.dir/build.make CMakeFiles/ROSBUILD_genmsg_lisp.dir/build
.PHONY : ROSBUILD_genmsg_lisp/fast

#=============================================================================
# Target rules for targets named ROSBUILD_gensrv_cpp

# Build rule for target.
ROSBUILD_gensrv_cpp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_gensrv_cpp
.PHONY : ROSBUILD_gensrv_cpp

# fast build rule for target.
ROSBUILD_gensrv_cpp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_gensrv_cpp.dir/build.make CMakeFiles/ROSBUILD_gensrv_cpp.dir/build
.PHONY : ROSBUILD_gensrv_cpp/fast

#=============================================================================
# Target rules for targets named ROSBUILD_gensrv_lisp

# Build rule for target.
ROSBUILD_gensrv_lisp: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ROSBUILD_gensrv_lisp
.PHONY : ROSBUILD_gensrv_lisp

# fast build rule for target.
ROSBUILD_gensrv_lisp/fast:
	$(MAKE) -f CMakeFiles/ROSBUILD_gensrv_lisp.dir/build.make CMakeFiles/ROSBUILD_gensrv_lisp.dir/build
.PHONY : ROSBUILD_gensrv_lisp/fast

#=============================================================================
# Target rules for targets named clean-test-results

# Build rule for target.
clean-test-results: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 clean-test-results
.PHONY : clean-test-results

# fast build rule for target.
clean-test-results/fast:
	$(MAKE) -f CMakeFiles/clean-test-results.dir/build.make CMakeFiles/clean-test-results.dir/build
.PHONY : clean-test-results/fast

#=============================================================================
# Target rules for targets named houghLineBasedTracker

# Build rule for target.
houghLineBasedTracker: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 houghLineBasedTracker
.PHONY : houghLineBasedTracker

# fast build rule for target.
houghLineBasedTracker/fast:
	$(MAKE) -f CMakeFiles/houghLineBasedTracker.dir/build.make CMakeFiles/houghLineBasedTracker.dir/build
.PHONY : houghLineBasedTracker/fast

#=============================================================================
# Target rules for targets named kltFbTracker

# Build rule for target.
kltFbTracker: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 kltFbTracker
.PHONY : kltFbTracker

# fast build rule for target.
kltFbTracker/fast:
	$(MAKE) -f CMakeFiles/kltFbTracker.dir/build.make CMakeFiles/kltFbTracker.dir/build
.PHONY : kltFbTracker/fast

#=============================================================================
# Target rules for targets named mbtEdgeTracker

# Build rule for target.
mbtEdgeTracker: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 mbtEdgeTracker
.PHONY : mbtEdgeTracker

# fast build rule for target.
mbtEdgeTracker/fast:
	$(MAKE) -f CMakeFiles/mbtEdgeTracker.dir/build.make CMakeFiles/mbtEdgeTracker.dir/build
.PHONY : mbtEdgeTracker/fast

#=============================================================================
# Target rules for targets named medianFlowTracking

# Build rule for target.
medianFlowTracking: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 medianFlowTracking
.PHONY : medianFlowTracking

# fast build rule for target.
medianFlowTracking/fast:
	$(MAKE) -f CMakeFiles/medianFlowTracking.dir/build.make CMakeFiles/medianFlowTracking.dir/build
.PHONY : medianFlowTracking/fast

#=============================================================================
# Target rules for targets named rosbuild_precompile

# Build rule for target.
rosbuild_precompile: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rosbuild_precompile
.PHONY : rosbuild_precompile

# fast build rule for target.
rosbuild_precompile/fast:
	$(MAKE) -f CMakeFiles/rosbuild_precompile.dir/build.make CMakeFiles/rosbuild_precompile.dir/build
.PHONY : rosbuild_precompile/fast

#=============================================================================
# Target rules for targets named rosbuild_premsgsrvgen

# Build rule for target.
rosbuild_premsgsrvgen: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rosbuild_premsgsrvgen
.PHONY : rosbuild_premsgsrvgen

# fast build rule for target.
rosbuild_premsgsrvgen/fast:
	$(MAKE) -f CMakeFiles/rosbuild_premsgsrvgen.dir/build.make CMakeFiles/rosbuild_premsgsrvgen.dir/build
.PHONY : rosbuild_premsgsrvgen/fast

#=============================================================================
# Target rules for targets named rospack_genmsg

# Build rule for target.
rospack_genmsg: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rospack_genmsg
.PHONY : rospack_genmsg

# fast build rule for target.
rospack_genmsg/fast:
	$(MAKE) -f CMakeFiles/rospack_genmsg.dir/build.make CMakeFiles/rospack_genmsg.dir/build
.PHONY : rospack_genmsg/fast

#=============================================================================
# Target rules for targets named rospack_genmsg_libexe

# Build rule for target.
rospack_genmsg_libexe: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rospack_genmsg_libexe
.PHONY : rospack_genmsg_libexe

# fast build rule for target.
rospack_genmsg_libexe/fast:
	$(MAKE) -f CMakeFiles/rospack_genmsg_libexe.dir/build.make CMakeFiles/rospack_genmsg_libexe.dir/build
.PHONY : rospack_genmsg_libexe/fast

#=============================================================================
# Target rules for targets named rospack_gensrv

# Build rule for target.
rospack_gensrv: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 rospack_gensrv
.PHONY : rospack_gensrv

# fast build rule for target.
rospack_gensrv/fast:
	$(MAKE) -f CMakeFiles/rospack_gensrv.dir/build.make CMakeFiles/rospack_gensrv.dir/build
.PHONY : rospack_gensrv/fast

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

#=============================================================================
# Target rules for targets named test-future

# Build rule for target.
test-future: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test-future
.PHONY : test-future

# fast build rule for target.
test-future/fast:
	$(MAKE) -f CMakeFiles/test-future.dir/build.make CMakeFiles/test-future.dir/build
.PHONY : test-future/fast

#=============================================================================
# Target rules for targets named test-results

# Build rule for target.
test-results: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test-results
.PHONY : test-results

# fast build rule for target.
test-results/fast:
	$(MAKE) -f CMakeFiles/test-results.dir/build.make CMakeFiles/test-results.dir/build
.PHONY : test-results/fast

#=============================================================================
# Target rules for targets named test-results-run

# Build rule for target.
test-results-run: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test-results-run
.PHONY : test-results-run

# fast build rule for target.
test-results-run/fast:
	$(MAKE) -f CMakeFiles/test-results-run.dir/build.make CMakeFiles/test-results-run.dir/build
.PHONY : test-results-run/fast

#=============================================================================
# Target rules for targets named tests

# Build rule for target.
tests: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 tests
.PHONY : tests

# fast build rule for target.
tests/fast:
	$(MAKE) -f CMakeFiles/tests.dir/build.make CMakeFiles/tests.dir/build
.PHONY : tests/fast

#=============================================================================
# Target rules for targets named track_node

# Build rule for target.
track_node: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 track_node
.PHONY : track_node

# fast build rule for target.
track_node/fast:
	$(MAKE) -f CMakeFiles/track_node.dir/build.make CMakeFiles/track_node.dir/build
.PHONY : track_node/fast

#=============================================================================
# Target rules for targets named track_ros

# Build rule for target.
track_ros: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 track_ros
.PHONY : track_ros

# fast build rule for target.
track_ros/fast:
	$(MAKE) -f CMakeFiles/track_ros.dir/build.make CMakeFiles/track_ros.dir/build
.PHONY : track_ros/fast

# target to build an object file
src/CamShiftTracking.o:
	$(MAKE) -f CMakeFiles/CamShiftTracking.dir/build.make CMakeFiles/CamShiftTracking.dir/src/CamShiftTracking.o
.PHONY : src/CamShiftTracking.o

# target to preprocess a source file
src/CamShiftTracking.i:
	$(MAKE) -f CMakeFiles/CamShiftTracking.dir/build.make CMakeFiles/CamShiftTracking.dir/src/CamShiftTracking.i
.PHONY : src/CamShiftTracking.i

# target to generate assembly for a file
src/CamShiftTracking.s:
	$(MAKE) -f CMakeFiles/CamShiftTracking.dir/build.make CMakeFiles/CamShiftTracking.dir/src/CamShiftTracking.s
.PHONY : src/CamShiftTracking.s

# target to build an object file
src/KernelBasedTracking.o:
	$(MAKE) -f CMakeFiles/KernelBasedTracking.dir/build.make CMakeFiles/KernelBasedTracking.dir/src/KernelBasedTracking.o
.PHONY : src/KernelBasedTracking.o

# target to preprocess a source file
src/KernelBasedTracking.i:
	$(MAKE) -f CMakeFiles/KernelBasedTracking.dir/build.make CMakeFiles/KernelBasedTracking.dir/src/KernelBasedTracking.i
.PHONY : src/KernelBasedTracking.i

# target to generate assembly for a file
src/KernelBasedTracking.s:
	$(MAKE) -f CMakeFiles/KernelBasedTracking.dir/build.make CMakeFiles/KernelBasedTracking.dir/src/KernelBasedTracking.s
.PHONY : src/KernelBasedTracking.s

# target to build an object file
src/houghLineBasedTracker.o:
	$(MAKE) -f CMakeFiles/houghLineBasedTracker.dir/build.make CMakeFiles/houghLineBasedTracker.dir/src/houghLineBasedTracker.o
.PHONY : src/houghLineBasedTracker.o

# target to preprocess a source file
src/houghLineBasedTracker.i:
	$(MAKE) -f CMakeFiles/houghLineBasedTracker.dir/build.make CMakeFiles/houghLineBasedTracker.dir/src/houghLineBasedTracker.i
.PHONY : src/houghLineBasedTracker.i

# target to generate assembly for a file
src/houghLineBasedTracker.s:
	$(MAKE) -f CMakeFiles/houghLineBasedTracker.dir/build.make CMakeFiles/houghLineBasedTracker.dir/src/houghLineBasedTracker.s
.PHONY : src/houghLineBasedTracker.s

# target to build an object file
src/kltFbTracker.o:
	$(MAKE) -f CMakeFiles/kltFbTracker.dir/build.make CMakeFiles/kltFbTracker.dir/src/kltFbTracker.o
.PHONY : src/kltFbTracker.o

# target to preprocess a source file
src/kltFbTracker.i:
	$(MAKE) -f CMakeFiles/kltFbTracker.dir/build.make CMakeFiles/kltFbTracker.dir/src/kltFbTracker.i
.PHONY : src/kltFbTracker.i

# target to generate assembly for a file
src/kltFbTracker.s:
	$(MAKE) -f CMakeFiles/kltFbTracker.dir/build.make CMakeFiles/kltFbTracker.dir/src/kltFbTracker.s
.PHONY : src/kltFbTracker.s

# target to build an object file
src/mbtEdgeTracker.o:
	$(MAKE) -f CMakeFiles/mbtEdgeTracker.dir/build.make CMakeFiles/mbtEdgeTracker.dir/src/mbtEdgeTracker.o
.PHONY : src/mbtEdgeTracker.o

# target to preprocess a source file
src/mbtEdgeTracker.i:
	$(MAKE) -f CMakeFiles/mbtEdgeTracker.dir/build.make CMakeFiles/mbtEdgeTracker.dir/src/mbtEdgeTracker.i
.PHONY : src/mbtEdgeTracker.i

# target to generate assembly for a file
src/mbtEdgeTracker.s:
	$(MAKE) -f CMakeFiles/mbtEdgeTracker.dir/build.make CMakeFiles/mbtEdgeTracker.dir/src/mbtEdgeTracker.s
.PHONY : src/mbtEdgeTracker.s

# target to build an object file
src/medianFlowTracking.o:
	$(MAKE) -f CMakeFiles/medianFlowTracking.dir/build.make CMakeFiles/medianFlowTracking.dir/src/medianFlowTracking.o
.PHONY : src/medianFlowTracking.o

# target to preprocess a source file
src/medianFlowTracking.i:
	$(MAKE) -f CMakeFiles/medianFlowTracking.dir/build.make CMakeFiles/medianFlowTracking.dir/src/medianFlowTracking.i
.PHONY : src/medianFlowTracking.i

# target to generate assembly for a file
src/medianFlowTracking.s:
	$(MAKE) -f CMakeFiles/medianFlowTracking.dir/build.make CMakeFiles/medianFlowTracking.dir/src/medianFlowTracking.s
.PHONY : src/medianFlowTracking.s

# target to build an object file
src/track_node.o:
	$(MAKE) -f CMakeFiles/track_node.dir/build.make CMakeFiles/track_node.dir/src/track_node.o
.PHONY : src/track_node.o

# target to preprocess a source file
src/track_node.i:
	$(MAKE) -f CMakeFiles/track_node.dir/build.make CMakeFiles/track_node.dir/src/track_node.i
.PHONY : src/track_node.i

# target to generate assembly for a file
src/track_node.s:
	$(MAKE) -f CMakeFiles/track_node.dir/build.make CMakeFiles/track_node.dir/src/track_node.s
.PHONY : src/track_node.s

# target to build an object file
src/track_ros.o:
	$(MAKE) -f CMakeFiles/track_ros.dir/build.make CMakeFiles/track_ros.dir/src/track_ros.o
.PHONY : src/track_ros.o

# target to preprocess a source file
src/track_ros.i:
	$(MAKE) -f CMakeFiles/track_ros.dir/build.make CMakeFiles/track_ros.dir/src/track_ros.i
.PHONY : src/track_ros.i

# target to generate assembly for a file
src/track_ros.s:
	$(MAKE) -f CMakeFiles/track_ros.dir/build.make CMakeFiles/track_ros.dir/src/track_ros.s
.PHONY : src/track_ros.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... CamShiftTracking"
	@echo "... KernelBasedTracking"
	@echo "... ROSBUILD_genmsg_cpp"
	@echo "... ROSBUILD_genmsg_lisp"
	@echo "... ROSBUILD_gensrv_cpp"
	@echo "... ROSBUILD_gensrv_lisp"
	@echo "... clean-test-results"
	@echo "... edit_cache"
	@echo "... houghLineBasedTracker"
	@echo "... kltFbTracker"
	@echo "... mbtEdgeTracker"
	@echo "... medianFlowTracking"
	@echo "... rebuild_cache"
	@echo "... rosbuild_precompile"
	@echo "... rosbuild_premsgsrvgen"
	@echo "... rospack_genmsg"
	@echo "... rospack_genmsg_libexe"
	@echo "... rospack_gensrv"
	@echo "... test"
	@echo "... test-future"
	@echo "... test-results"
	@echo "... test-results-run"
	@echo "... tests"
	@echo "... track_node"
	@echo "... track_ros"
	@echo "... src/CamShiftTracking.o"
	@echo "... src/CamShiftTracking.i"
	@echo "... src/CamShiftTracking.s"
	@echo "... src/KernelBasedTracking.o"
	@echo "... src/KernelBasedTracking.i"
	@echo "... src/KernelBasedTracking.s"
	@echo "... src/houghLineBasedTracker.o"
	@echo "... src/houghLineBasedTracker.i"
	@echo "... src/houghLineBasedTracker.s"
	@echo "... src/kltFbTracker.o"
	@echo "... src/kltFbTracker.i"
	@echo "... src/kltFbTracker.s"
	@echo "... src/mbtEdgeTracker.o"
	@echo "... src/mbtEdgeTracker.i"
	@echo "... src/mbtEdgeTracker.s"
	@echo "... src/medianFlowTracking.o"
	@echo "... src/medianFlowTracking.i"
	@echo "... src/medianFlowTracking.s"
	@echo "... src/track_node.o"
	@echo "... src/track_node.i"
	@echo "... src/track_node.s"
	@echo "... src/track_ros.o"
	@echo "... src/track_ros.i"
	@echo "... src/track_ros.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

