cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER clang)

project(ffnn LANGUAGES C)


set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")


# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


# Debug configuration with sanitizers
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address -fsanitize=undefined")


# Release configuration
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "")

# Collect source files
file(GLOB C_DS_LIB "external/C-Data-Structures-Lib/src/*.c")
file(GLOB C_DS_INCLUDE "external/C-Data-Structures-Lib/include")


# Core library sources (excluding main.c)
set(CORE_SRC_FILES
    src/ffnn.c
    src/idx_file_reader.c
    src/layer.c
    src/mnist_data_processor.c
)

# GUI-specific raylib sources
set(GUI_RAYLIB_FILES
    external/raylib/src/mnist_predictor.c
    external/raylib/src/ray_main.c
)

# Define raylib location
set(RAYLIB_DIR ${CMAKE_BINARY_DIR}/raylib)
set(RAYLIB_LIB ${RAYLIB_DIR}/lib/libraylib.so)
set(RAYLIB_INCLUDE_DIR ${RAYLIB_DIR}/include)

# Verify raylib exists
if(NOT EXISTS ${RAYLIB_LIB})
    message(FATAL_ERROR "raylib not found at: ${RAYLIB_LIB}")
else()
    message(STATUS "raylib found at: ${RAYLIB_LIB}")
endif()


# Main executable (CLI)
add_executable(main src/main.c ${CORE_SRC_FILES} ${C_DS_LIB})

# GUI executable
add_executable(gui ${CORE_SRC_FILES} ${C_DS_LIB} ${GUI_RAYLIB_FILES})

target_include_directories(main PRIVATE 
                                include 
                                external/raylib/include
                                tests
                                ${C_DS_INCLUDE}
)

target_include_directories(gui PRIVATE 
                                include 
                                external/raylib/include
                                tests
                                ${C_DS_INCLUDE}
                                ${RAYLIB_INCLUDE_DIR}
)

# Link raylib to GUI only
target_link_libraries(gui PRIVATE ${RAYLIB_LIB})


