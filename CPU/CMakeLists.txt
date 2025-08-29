cmake_minimum_required(VERSION 3.14)
project(CPU)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O1 -march=native)

include_directories(.)
include_directories(Common)
include_directories(Struct)
include_directories(Src)

add_executable(CPU main.cpp Common/murmur3.c)