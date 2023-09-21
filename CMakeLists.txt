
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")

project(test)

message("Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

find_package(HDF5 REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(test main.cpp )
target_link_libraries(test ${MPI_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} ${HDF5_CXX_LIBRARIES}) 

