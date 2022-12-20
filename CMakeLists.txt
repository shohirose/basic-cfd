cmake_minimum_required(VERSION 3.13)

include(FetchContent)
FetchContent_Declare(
  Eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  )
set(EIGEN_BUILD_DOC OFF FORCE)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
set(BUILD_TESTING OFF FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF FORCE)

FetchContent_MakeAvailable(Eigen)

project(basic_cfd CXX)

add_library(cfd
    src/ftcs_scheme.cpp
    src/lax_scheme.cpp
    src/lax_wendroff_scheme.cpp
    src/upwind_scheme.cpp
  )
target_include_directories(cfd PUBLIC include/)
target_link_libraries(cfd PUBLIC Eigen3::Eigen)
target_compile_features(cfd PUBLIC cxx_std_17)
target_compile_options(cfd
  PUBLIC
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /arch:AVX512>
  )

add_executable(advection_1d examples/advection_1d.cpp)
target_link_libraries(advection_1d PRIVATE cfd)