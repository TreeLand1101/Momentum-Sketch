cmake_minimum_required(VERSION 3.14)
project(FilterPerformance)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O1 -mavx2 -mfma -mbmi2 -march=native)

include_directories(../../Common)
include_directories(../../Struct)

add_executable(FilterPerformance FilterPerformance.cpp  ../../Common/murmur3.c)