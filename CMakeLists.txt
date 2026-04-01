cmake_minimum_required(VERSION 3.19.0)
project(gauss_splat)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/kernels)
add_subdirectory(${PROJECT_SOURCE_DIR}/gauss_splat/csrc)