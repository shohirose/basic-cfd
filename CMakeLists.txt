cmake_minimum_required(VERSION 3.13)

# ----------------------------- Eigen library ----------------------------------
include(FetchContent)
FetchContent_Declare(
  Eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  )
set(EIGEN_BUILD_DOC OFF FORCE)
#set(BUILD_TESTING OFF FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF FORCE)
FetchContent_MakeAvailable(Eigen)

# ------------------------------ fmt library -----------------------------------
FetchContent_Declare(
  fmt
  URL https://github.com/fmtlib/fmt/releases/download/9.1.0/fmt-9.1.0.zip
  )
FetchContent_MakeAvailable(fmt)

project(basic_cfd CXX)

add_library(cfd
  INTERFACE
    include/ftcs_scheme_1d.hpp
    include/lax_scheme_1d.hpp
    include/lax_wendroff_scheme_1d.hpp
    include/first_order_upwind_scheme_1d.hpp
    include/second_order_upwind_scheme_1d.hpp
    include/harten_yee_tvd_scheme_1d.hpp
    include/muscl_tvd_scheme_1d.hpp
    include/advection_equation_solver_1d.hpp
    include/file_writer.hpp
  )
target_include_directories(cfd INTERFACE include/)
target_link_libraries(cfd
  INTERFACE
    Eigen3::Eigen
    fmt::fmt
  )
target_compile_features(cfd INTERFACE cxx_std_17)
target_compile_options(cfd
  INTERFACE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
    $<$<CXX_COMPILER_ID:Intel>:$<IF:$<PLATFORM_ID:Windows>,/W3,-w3>>
    $<$<CXX_COMPILER_ID:IntelLLVM>:$<IF:$<PLATFORM_ID:Windows>,/W3,-w3>>
  )

add_executable(advection_1d src/advection_1d.cpp)
target_link_libraries(advection_1d
  PRIVATE
    cfd    
  )