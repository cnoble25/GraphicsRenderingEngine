# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2024.3.1\bin\cmake\win\x64\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2024.3.1\bin\cmake\win\x64\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\carso\CLionProjects\GraphicsRenderer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\carso\CLionProjects\GraphicsRenderer\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/GraphicsRenderer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GraphicsRenderer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GraphicsRenderer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GraphicsRenderer.dir/flags.make

CMakeFiles/GraphicsRenderer.dir/main.cpp.obj: CMakeFiles/GraphicsRenderer.dir/flags.make
CMakeFiles/GraphicsRenderer.dir/main.cpp.obj: C:/Users/carso/CLionProjects/GraphicsRenderer/main.cpp
CMakeFiles/GraphicsRenderer.dir/main.cpp.obj: CMakeFiles/GraphicsRenderer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\carso\CLionProjects\GraphicsRenderer\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GraphicsRenderer.dir/main.cpp.obj"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GraphicsRenderer.dir/main.cpp.obj -MF CMakeFiles\GraphicsRenderer.dir\main.cpp.obj.d -o CMakeFiles\GraphicsRenderer.dir\main.cpp.obj -c C:\Users\carso\CLionProjects\GraphicsRenderer\main.cpp

CMakeFiles/GraphicsRenderer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GraphicsRenderer.dir/main.cpp.i"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\carso\CLionProjects\GraphicsRenderer\main.cpp > CMakeFiles\GraphicsRenderer.dir\main.cpp.i

CMakeFiles/GraphicsRenderer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GraphicsRenderer.dir/main.cpp.s"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\carso\CLionProjects\GraphicsRenderer\main.cpp -o CMakeFiles\GraphicsRenderer.dir\main.cpp.s

# Object files for target GraphicsRenderer
GraphicsRenderer_OBJECTS = \
"CMakeFiles/GraphicsRenderer.dir/main.cpp.obj"

# External object files for target GraphicsRenderer
GraphicsRenderer_EXTERNAL_OBJECTS =

GraphicsRenderer.exe: CMakeFiles/GraphicsRenderer.dir/main.cpp.obj
GraphicsRenderer.exe: CMakeFiles/GraphicsRenderer.dir/build.make
GraphicsRenderer.exe: CMakeFiles/GraphicsRenderer.dir/linkLibs.rsp
GraphicsRenderer.exe: CMakeFiles/GraphicsRenderer.dir/objects1.rsp
GraphicsRenderer.exe: CMakeFiles/GraphicsRenderer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\carso\CLionProjects\GraphicsRenderer\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable GraphicsRenderer.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\GraphicsRenderer.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GraphicsRenderer.dir/build: GraphicsRenderer.exe
.PHONY : CMakeFiles/GraphicsRenderer.dir/build

CMakeFiles/GraphicsRenderer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\GraphicsRenderer.dir\cmake_clean.cmake
.PHONY : CMakeFiles/GraphicsRenderer.dir/clean

CMakeFiles/GraphicsRenderer.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\carso\CLionProjects\GraphicsRenderer C:\Users\carso\CLionProjects\GraphicsRenderer C:\Users\carso\CLionProjects\GraphicsRenderer\cmake-build-debug C:\Users\carso\CLionProjects\GraphicsRenderer\cmake-build-debug C:\Users\carso\CLionProjects\GraphicsRenderer\cmake-build-debug\CMakeFiles\GraphicsRenderer.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/GraphicsRenderer.dir/depend

