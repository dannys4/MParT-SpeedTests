cmake_minimum_required (VERSION 3.13)

project(SimpleExample)

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-deprecated-gpu-targets" COMPILER_IS_NVCC1)
check_cxx_compiler_flag("--expt-relaxed-constexpr" COMPILER_IS_NVCC2)
if(COMPILER_IS_NVCC1 AND COMPILER_IS_NVCC2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr")
    add_compile_definitions(EIGEN_NO_CUDA)
endif()


find_package(Kokkos REQUIRED)
find_package(MParT REQUIRED)
message(STATUS "KOKKOS_FOUND = ${Kokkos_FOUND}")
message(STATUS "MPART_FOUND = ${MParT_FOUND}")

add_executable(SampleSpeed SpeedTests_samples.cpp)
target_link_libraries(SampleSpeed MParT::mpart Kokkos::kokkos Eigen3::Eigen)
