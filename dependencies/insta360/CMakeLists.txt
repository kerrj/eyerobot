cmake_minimum_required(VERSION 3.15)
project(insta360)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add pybind11 directly instead of finding it
add_subdirectory(pybind11)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Define camera SDK paths - you'll need to adjust these
set(CAMERA_SDK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/CameraSDK/include")
set(CAMERA_SDK_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/CameraSDK/lib")

# Include directories
include_directories(${CAMERA_SDK_INCLUDE_DIR})

# Create the Python module
pybind11_add_module(_camera_sdk src/camera_module.cpp)

# Link with the Camera SDK libraries
if(WIN32)
    target_link_libraries(_camera_sdk PRIVATE ${CAMERA_SDK_LIB_DIR}/camera.lib)
elseif(APPLE)
    target_link_libraries(_camera_sdk PRIVATE ${CAMERA_SDK_LIB_DIR}/libcamera.dylib)
else()
    target_link_libraries(_camera_sdk PRIVATE ${CAMERA_SDK_LIB_DIR}/libCameraSDK.so)
endif()