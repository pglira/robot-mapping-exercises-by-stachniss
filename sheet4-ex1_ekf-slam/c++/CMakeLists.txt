cmake_minimum_required(VERSION 3.15) # was 3.16
project(ekf-slam)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(autodiff)

add_executable(ekf-slam main.cpp main.h)

target_link_libraries (ekf-slam Eigen3::Eigen autodiff::autodiff)
