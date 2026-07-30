// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptoas.h"

#include "PTOModule.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <string>
#include <vector>

namespace py = pybind11;

namespace {

int runPTOASFromPython(const std::vector<std::string> &arguments) {
  std::vector<std::string> storage = arguments;
  std::vector<char *> argv;
  argv.reserve(storage.size());
  for (std::string &argument : storage)
    argv.push_back(argument.data());

  py::gil_scoped_release release;
  return mlir::pto::runPTOAS(static_cast<int>(argv.size()), argv.data());
}

} // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "PTOAS compiler and PTO dialect native bindings";
  py::module_::import("ptoas.mlir.ir");
  mlir::pto::python::populatePTODialectBindings(module);
  module.def("main", &runPTOASFromPython, py::arg("argv"));
}
