// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.h - PTO Dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This file defines the dialect for the PTO Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTO_H_
#define MLIR_DIALECT_PTO_IR_PTO_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// PTO Dialect
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTODialect.h"

//===----------------------------------------------------------------------===//
// PTO Enums
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOEnums.h.inc"

//===----------------------------------------------------------------------===//
// PTO Interfaces
//===----------------------------------------------------------------------===//
 
#include "PTO/IR/PTOInterfaces.h.inc"
#include "PTO/IR/VPTOInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// PTO Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.h.inc"

//===----------------------------------------------------------------------===//
// PTO Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.h.inc"

//===----------------------------------------------------------------------===//
// PTO Dialect Operations
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pto {

//===----------------------------------------------------------------------===//
// S Fractal Size Constants
//===----------------------------------------------------------------------===//

/// Fractal size for mxBox layout (16x2 inner block, 32 bytes total).
inline constexpr int32_t kFractalMxSize = 32;

/// Fractal size for AB matrices in matmul (16xN inner block, 512 bytes).
inline constexpr int32_t kFractalABSize = 512;

/// Fractal size for C matrix in matmul (16x16 inner block, 1024 bytes).
inline constexpr int32_t kFractalCSize = 1024;

struct DmaLoopConfig {
  Value count;
  Value srcStride;
  Value dstStride;
};

struct DmaPadConfig {
  Value value;
  Value leftCount;
  Value rightCount;
};

struct AccStoreModeConfig {
  AccStoreMode mode;
  std::optional<Value> split;
  std::optional<Value> loop0SrcStride;
};

struct CubeLoadFracShapeConfig {
  Value nValue;
  Value dValue;
};

struct CubeLoadFracSrcLayoutConfig {
  Value srcInnerStride;
  std::optional<Value> srcOuterStride;
};

struct CubeLoadFracDstGroupConfig {
  Value groupCount;
  Value dstLoop2Stride;
  Value dstLoop3Stride;
  Value dstLoop4Stride;
};

struct CubeLoadFracCtrlConfig {
  Value l2CacheCtrl;
  Value smallc0En;
};

} // namespace pto
} // namespace mlir

#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.h.inc"

namespace mlir {
class MLIRContext;
class TypeConverter;

namespace pto {

inline constexpr char kPTOTargetArchAttrName[] = "pto.target_arch";

/// Get PTO Address Space Attr from input type.
AddressSpaceAttr getPTOAddressSpaceAttr(Type type);

/// Return true if type is a ptr/memref in GM address space (or default).
bool isScalarPtrOrMemRef(Type type);

enum class PTOArch {
  A3,
  A5,
};

/// Resolve the effective PTO target architecture from module-level IR state.
PTOArch getTargetArch(ModuleOp module);
PTOArch getTargetArch(Operation *op);
bool isTargetArchA3(ModuleOp module);
bool isTargetArchA5(ModuleOp module);
bool isTargetArchA3(Operation *op);
bool isTargetArchA5(Operation *op);

enum class PTOParserTargetArch {
  Unspecified,
  A3,
  A5,
};

void setPTOParserTargetArch(MLIRContext *context, PTOParserTargetArch arch);
PTOParserTargetArch getPTOParserTargetArch(MLIRContext *context);

class ScopedPTOParserTargetArch {
public:
  explicit ScopedPTOParserTargetArch(MLIRContext *context,
                                     PTOParserTargetArch arch);
  ~ScopedPTOParserTargetArch();

private:
  MLIRContext *context;
  PTOParserTargetArch previousArch;
};


/// Function attributes that mark an explicit PTO kernel entry.
inline constexpr llvm::StringLiteral kPTOEntryAttrName = "pto.entry";
inline constexpr llvm::StringLiteral kLegacyHACCEntryAttrName = "hacc.entry";
inline constexpr llvm::StringLiteral kPTOKernelAttrName = "pto.kernel";
inline constexpr llvm::StringLiteral kLegacyPTOAICoreAttrName = "pto.aicore";
inline constexpr llvm::StringLiteral kPTOSimtEntryAttrName = "pto.simt_entry";
inline constexpr llvm::StringLiteral kPTOSimtMaxThreadsAttrName =
    "pto.simt_max_threads";
inline constexpr llvm::StringLiteral kPTOSimtMaxRegistersAttrName =
    "pto.simt_max_regs";

/// Return true if the function carries an explicit entry marker. PTO accepts
/// both the EmitC naming (`pto.entry`) and VPTO naming (`pto.kernel`) as entry
/// aliases; `hacc.entry` and `pto.aicore` are legacy aliases.
bool hasExplicitPTOEntryAttr(func::FuncOp func);

/// Return true if the function should be emitted as an AICORE entry.
bool isPTOEntryFunction(func::FuncOp func);

/// Validate module-level PTO entry configuration before EmitC lowering.
LogicalResult validatePTOEntryFunctions(ModuleOp module);

/// Clear internal PTO entry selection markers from function attributes.
void annotatePTOEntryFunctions(ModuleOp module);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTO_H_
