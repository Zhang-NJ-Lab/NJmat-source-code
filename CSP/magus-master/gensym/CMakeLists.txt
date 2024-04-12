cmake_minimum_required(VERSION 3.15)
project(gensym LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(../pybind11 pybind11_binary)
#add_definitions(-DDEBUG)
pybind11_add_module(gensym src/main.cpp)
