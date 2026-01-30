cmake_minimum_required(VERSION 3.20.0)
project(ptoas)

# =========================================================
# [新增] 强制设置 C++17 标准 (LLVM 19 必需)
# =========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================================================
# 1. 显式寻找 LLVM (这是解决 tablegen 报错的关键)
# =========================================================
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM CMake Dir: ${LLVM_CMAKE_DIR}")
message(STATUS "LLVM Include Dir: ${LLVM_INCLUDE_DIRS}")

# 将 LLVM 模块路径加入 CMake 搜索路径
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# 关键：加载 LLVM 基础宏 (这里定义了 tablegen, add_llvm_pass 等)
include(AddLLVM)
include(TableGen) # 双重保险，确保 tablegen 宏可用
include(HandleLLVMOptions) # 处理 RTTI, EH 等编译选项

# =========================================================
# 2. 寻找 MLIR
# =========================================================
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "MLIR CMake Dir: ${MLIR_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# =========================================================
# 3. 配置构建环境
# =========================================================
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# =========================================================
# 4. 添加子目录
# =========================================================
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# 开启 Python 绑定选项
option(PTO_ENABLE_PYTHON_BINDING "Enable Python bindings" ON)

if(PTO_ENABLE_PYTHON_BINDING)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  # 设置 Python 扩展的输出目录，方便调试
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pto)

  add_subdirectory(python)
endif()
