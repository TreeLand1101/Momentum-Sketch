cmake_minimum_required(VERSION 3.14)
project(PacketCountDistribution)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O1 -march=native)

include_directories(../../Common)
include_directories(../../Struct)

add_executable(PacketCountDistribution PacketCountDistribution.cpp  ../../Common/murmur3.c)