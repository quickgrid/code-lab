cmake_minimum_required(VERSION 3.12.0)
project(myapp VERSION 0.1.0)
add_executable(myapp1 src/memory_leak.cpp)
add_executable(myapp2 src/stack_frame_debug.cpp)
