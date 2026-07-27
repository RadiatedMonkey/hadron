cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

project(hadron)

include(cmake/CPM.cmake)

set(FETCHCONTENT_QUIET  FALSE CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 20)

if(WIN32)
  set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()

set(SLANG_ENABLE_SLANGC         OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_SLANGD         OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_SLANGI         OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_SLANG_GLSLANG  OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_SLANGRT        OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_GFX            OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_TESTS          OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_DXIL           OFF CACHE BOOL "" FORCE)
set(SLANG_ENABLE_EXAMPLES       OFF CACHE BOOL "" FORCE)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

CPMAddPackage("gh:GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator#3aa9212@v3.4.0")
CPMAddPackage("gh:gabime/spdlog#79524dd@v1.17.0")
CPMAddPackage("gh:shader-slang/slang#4265906@v2026.14")
CPMAddPackage("gh:glfw/glfw#7b6aead@3.4")
CPMAddPackage("gh:zeux/volk#3ca312a@1.4.350")

set(SOURCES
    src/main.cpp
    src/vulkan/instance.cpp
    src/vulkan/device.cpp
    src/vulkan/pipeline.cpp
    src/vulkan/storage.cpp
    src/vulkan/command.cpp
    src/vulkan/fence.cpp
    src/vulkan/shader.cpp

    src/workloads/rand.cpp
    src/workloads/stats.cpp
)

add_executable(hadron ${SOURCES})
target_include_directories(hadron PRIVATE include)
target_compile_definitions(hadron PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(hadron PRIVATE volk glfw spdlog slang GPUOpen::VulkanMemoryAllocator)
