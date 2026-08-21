cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)
project(rmsnorm_row_vf_matrix LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(PTO_ISA_ROOT "" CACHE PATH "Path to PTO-ISA")
set(GENERATED_ROOT "" CACHE PATH "Path to generated PTOAS objects")
set(ASCENDC_SOURCE "" CACHE FILEPATH "AscendC RMSNorm source")
set(RMSNORM_ROWS "" CACHE STRING "Static row count")
if(NOT PTO_ISA_ROOT OR NOT GENERATED_ROOT OR NOT ASCENDC_SOURCE OR NOT RMSNORM_ROWS)
  message(FATAL_ERROR "PTO_ISA_ROOT, GENERATED_ROOT, ASCENDC_SOURCE and RMSNORM_ROWS are required")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
include_directories(
  ${PTO_ISA_ROOT}/include
  ${PTO_ISA_ROOT}/tests/common
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/x86_64-linux/asc
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/include
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/include/basic_api
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/include/interface
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/impl
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/impl/basic_api
  ${ASCEND_DRIVER_PATH}/kernel/inc)

set(CCE_BASE_OPTIONS
  -xcce -fenable-matrix --cce-aicore-enable-tl
  --cce-aicore-arch=dav-c310-vec
  -DREGISTER_BASE -D__DAV_VEC__ -fPIC -O2 -std=c++17
  -Wno-macro-redefined -Wno-ignored-attributes
  -Xhost-start -Xhost-end
  "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-record-overflow=true"
  "SHELL:-mllvm -cce-aicore-addr-transform"
  "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")

set(VF_OFF_OPTIONS
  "SHELL:-mllvm -cce-vf-enable-vf-fusion=false"
  "SHELL:-mllvm -cce-vf-enable-loop-fusion=false"
  "SHELL:-mllvm -cce-vf-enable-vf-ldst-elimination=false"
  "SHELL:-mllvm -cce-vf-enable-ub-dead-st-elimination=false"
  "SHELL:-mllvm -cce-vf-auto-sync=off"
  "SHELL:-mllvm -cce-vf-remove-membar=false")

function(add_host_runner target kernel_target launch_name)
  add_executable(${target} common/main.cpp)
  target_compile_options(${target} PRIVATE
    -xc++ -O2 -std=c++17 "SHELL:-include stdint.h"
    "SHELL:-include stddef.h")
  target_compile_definitions(${target} PRIVATE
    RMSNORM_ROWS=${RMSNORM_ROWS}
    RMSNORM_LAUNCH_NAME=${launch_name})
  target_link_directories(${target} PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/x86_64-linux/lib64
    ${ASCEND_HOME_PATH}/aarch64-linux/lib64)
  find_library(NNOPBASE_LIB NAMES nnopbase HINTS
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/x86_64-linux/lib64
    ${ASCEND_HOME_PATH}/aarch64-linux/lib64)
  target_link_libraries(${target} PRIVATE ${kernel_target} runtime stdc++
    ascendcl m tiling_api platform c_sec dl ${NNOPBASE_LIB})
  target_link_options(${target} PRIVATE -Wl,--allow-shlib-undefined)
endfunction()

function(add_vpto_variant variant kernel_name launch_name)
  set(fat_object ${GENERATED_ROOT}/${variant}.fat.o)
  set_source_files_properties(${fat_object} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  add_library(${variant}_kernel SHARED ${fat_object} common/launch.cpp)
  target_compile_options(${variant}_kernel PRIVATE ${CCE_BASE_OPTIONS} ${VF_OFF_OPTIONS})
  target_compile_definitions(${variant}_kernel PRIVATE
    RMSNORM_KERNEL_NAME=${kernel_name}
    RMSNORM_LAUNCH_NAME=${launch_name})
  target_link_options(${variant}_kernel PRIVATE --cce-fatobj-link)
  add_host_runner(${variant} ${variant}_kernel ${launch_name})
endfunction()

function(add_ascendc_variant variant kernel_name launch_name use_vf)
  add_library(${variant}_kernel SHARED ${ASCENDC_SOURCE} common/launch.cpp)
  target_compile_options(${variant}_kernel PRIVATE ${CCE_BASE_OPTIONS} ${VF_OFF_OPTIONS})
  target_compile_definitions(${variant}_kernel PRIVATE
    RMSNORM_ROWS=${RMSNORM_ROWS}
    RMSNORM_KERNEL_NAME=${kernel_name}
    RMSNORM_LAUNCH_NAME=${launch_name})
  if(use_vf)
    target_compile_definitions(${variant}_kernel PRIVATE RMSNORM_USE_VF=1)
  endif()
  target_link_options(${variant}_kernel PRIVATE --cce-fatobj-link)
  add_host_runner(${variant} ${variant}_kernel ${launch_name})
endfunction()

foreach(variant IN ITEMS A B C D)
  if(NOT DEFINED ${variant}_KERNEL_NAME OR NOT DEFINED ${variant}_LAUNCH_NAME)
    message(FATAL_ERROR "${variant}_KERNEL_NAME and ${variant}_LAUNCH_NAME are required")
  endif()
  add_vpto_variant(${variant} ${${variant}_KERNEL_NAME} ${${variant}_LAUNCH_NAME})
endforeach()

foreach(variant IN ITEMS ACU ACF)
  if(NOT DEFINED ${variant}_KERNEL_NAME OR NOT DEFINED ${variant}_LAUNCH_NAME)
    message(FATAL_ERROR "${variant}_KERNEL_NAME and ${variant}_LAUNCH_NAME are required")
  endif()
endforeach()
add_ascendc_variant(ACU ${ACU_KERNEL_NAME} ${ACU_LAUNCH_NAME} FALSE)
add_ascendc_variant(ACF ${ACF_KERNEL_NAME} ${ACF_LAUNCH_NAME} TRUE)
