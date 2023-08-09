cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")

project(dpx LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52")

add_executable (dpx
    src/dpx.cu
    src/dpx.cuh
    src/main.cpp)

