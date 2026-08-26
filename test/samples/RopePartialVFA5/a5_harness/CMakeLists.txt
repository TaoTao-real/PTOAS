cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)
project(rope_partial_vf_matrix LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(PTO_ISA_ROOT "" CACHE PATH "Path to PTO-ISA")
set(GENERATED_ROOT "" CACHE PATH "Path to generated PTOAS objects")
set(MANUAL_VF_DIR "" CACHE PATH "Directory containing vf_rope_attached.h")
if(NOT PTO_ISA_ROOT OR NOT GENERATED_ROOT OR NOT MANUAL_VF_DIR)
  message(FATAL_ERROR "PTO_ISA_ROOT, GENERATED_ROOT and MANUAL_VF_DIR are required")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
include_directories(
  ${PTO_ISA_ROOT}/include
  ${PTO_ISA_ROOT}/tests/common
  ${MANUAL_VF_DIR}
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/x86_64-linux/asc
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/include
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/include/basic_api
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/include/interface
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/impl
  ${ASCEND_HOME_PATH}/x86_64-linux/asc/impl/basic_api
  ${ASCEND_HOME_PATH}/aarch64-linux/asc
  ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
  ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/basic_api
  ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
  ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl
  ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
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
  target_compile_definitions(${target} PRIVATE ROPE_LAUNCH_NAME=${launch_name})
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

function(add_vpto_variant target object_name kernel_name launch_name)
  set(fat_object ${GENERATED_ROOT}/${object_name})
  set_source_files_properties(${fat_object} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  add_library(${target}_kernel SHARED ${fat_object} common/launch.cpp)
  target_compile_options(${target}_kernel PRIVATE ${CCE_BASE_OPTIONS} ${VF_OFF_OPTIONS})
  target_compile_definitions(${target}_kernel PRIVATE
    ROPE_KERNEL_NAME=${kernel_name} ROPE_LAUNCH_NAME=${launch_name})
  target_link_options(${target}_kernel PRIVATE --cce-fatobj-link)
  add_host_runner(${target} ${target}_kernel ${launch_name})
endfunction()

function(add_manual_vf_variant target kernel_name launch_name interleave)
  add_library(${target}_kernel SHARED manual_kernel.cpp common/launch.cpp)
  target_compile_options(${target}_kernel PRIVATE ${CCE_BASE_OPTIONS} ${VF_OFF_OPTIONS})
  target_compile_definitions(${target}_kernel PRIVATE
    ROPE_KERNEL_NAME=${kernel_name} ROPE_LAUNCH_NAME=${launch_name})
  if(interleave)
    target_compile_definitions(${target}_kernel PRIVATE ROPE_MODE_INTERLEAVE=1)
  endif()
  target_link_options(${target}_kernel PRIVATE --cce-fatobj-link)
  add_host_runner(${target} ${target}_kernel ${launch_name})
endfunction()

add_vpto_variant(D_HALF half-D.fat.o rope_partial_half_n8
  launch_rope_partial_d_half)
add_vpto_variant(D_INTERLEAVE interleave-D.fat.o
  rope_partial_interleave_n8 launch_rope_partial_d_interleave)
add_manual_vf_variant(ACF_HALF rope_partial_acf_half
  launch_rope_partial_acf_half FALSE)
add_manual_vf_variant(ACF_INTERLEAVE rope_partial_acf_interleave
  launch_rope_partial_acf_interleave TRUE)
