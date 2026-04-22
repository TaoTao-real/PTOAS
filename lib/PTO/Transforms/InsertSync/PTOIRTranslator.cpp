// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Matchers.h"
// [P0 新增] 引入副作用接口和 PTO 接口
#include "mlir/Interfaces/SideEffectInterfaces.h"
 
#define DEBUG_TYPE "pto-ir-translator"
 
using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr const char *kLegacyMultiBufferAttr = "pto.multi_buffer";
constexpr const char *kSubviewMultiBufferFactorAttr =
    "pto.multi_buffer_factor";
constexpr const char *kSubviewMultiBufferSlotAttr =
    "pto.multi_buffer_slot";
constexpr const char *kSubviewMultiBufferGroupAttr =
    "pto.multi_buffer_group";

static bool isSupportedExplicitSubviewMultibufferFactor(int factor) {
  return factor > 1 &&
         factor < kMaxExplicitSubviewMultiBufferFactorExclusive;
}

static std::optional<int64_t> getConstantIntValue(Value v) {
  llvm::APInt apIntValue;
  if (matchPattern(v, m_ConstantInt(&apIntValue))) {
    return apIntValue.getSExtValue();
  }
  return std::nullopt;
}

static bool isSubviewLikeOp(Operation *op) {
  return isa<pto::SubViewOp, memref::SubViewOp>(op);
}

static bool isLegacyDoubleBufferRoot(const BaseMemInfo &info) {
  return info.baseAddresses.size() == 2 && info.allocateSize != 0 &&
         info.scope != pto::AddressSpace::GM;
}

static std::optional<uint64_t> getElementBytes(Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    uint64_t width = static_cast<uint64_t>((intType.getWidth() + 7) / 8);
    return width == 0 ? 1 : width;
  }
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    uint64_t width = static_cast<uint64_t>((floatType.getWidth() + 7) / 8);
    return width == 0 ? 1 : width;
  }
  return std::nullopt;
}

static std::optional<int64_t> accumulateStaticShapeBytes(ArrayRef<int64_t> shape,
                                                         Type elementType) {
  auto elemBytes = getElementBytes(elementType);
  if (!elemBytes)
    return std::nullopt;
  uint64_t numElems = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return std::nullopt;
    numElems *= static_cast<uint64_t>(dim);
  }
  return static_cast<int64_t>(numElems * *elemBytes);
}

static std::optional<int64_t> getStaticValueSizeInBytes(Value value) {
  if (!value)
    return std::nullopt;
  Type type = value.getType();
  if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
    return accumulateStaticShapeBytes(tileType.getShape(),
                                      tileType.getElementType());
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (!memrefType.hasStaticShape())
      return std::nullopt;
    return accumulateStaticShapeBytes(memrefType.getShape(),
                                      memrefType.getElementType());
  }
  return std::nullopt;
}

static bool areStaticBoxesOverlapping(ArrayRef<int64_t> aOffsets,
                                      ArrayRef<int64_t> aSizes,
                                      ArrayRef<int64_t> bOffsets,
                                      ArrayRef<int64_t> bSizes) {
  if (aOffsets.size() != bOffsets.size() || aSizes.size() != bSizes.size() ||
      aOffsets.size() != aSizes.size())
    return false;

  for (size_t i = 0; i < aOffsets.size(); ++i) {
    int64_t aBegin = aOffsets[i];
    int64_t aEnd = aBegin + aSizes[i];
    int64_t bBegin = bOffsets[i];
    int64_t bEnd = bBegin + bSizes[i];
    if (std::max(aBegin, bBegin) >= std::min(aEnd, bEnd))
      return false;
  }
  return true;
}

} // namespace

// [辅助函数] 尝试从 Operation 中计算相对于 Source 的字节偏移量和新大小
// 返回值: pair<offsetInBytes, sizeInBytes>
// 如果无法计算静态值，返回 {-1, -1} 表示这是动态的
static std::pair<int64_t, int64_t> getStaticOffsetAndSize(Operation *op, Value src) {
  auto srcType = dyn_cast<MemRefType>(src.getType());
  if (!srcType) return {0, 0};
  
  int64_t elemSize = srcType.getElementType().getIntOrFloatBitWidth() / 8;
  if (elemSize == 0) elemSize = 1;
 
  // === Case 1: memref.subview ===
  if (auto subView = dyn_cast<memref::SubViewOp>(op)) {
    int64_t baseOffset;
    SmallVector<int64_t, 4> strides;
    if (failed(mlir::getStridesAndOffset(srcType, strides, baseOffset))) {
        return {-1, -1}; 
    }
 
    int64_t newSize = 1;
    for (int64_t s : subView.getStaticSizes()) {
      if (s == ShapedType::kDynamic) return {-1, -1};
      newSize *= s;
    }
    newSize *= elemSize;
 
    int64_t totalOffset = 0;
    auto staticOffsets = subView.getStaticOffsets();
    
    if (staticOffsets.empty()) return {-1, -1};
    if (staticOffsets.size() > strides.size()) return {-1, -1}; 
 
    for (size_t i = 0; i < staticOffsets.size(); ++i) {
      int64_t off = staticOffsets[i];
      if (off == ShapedType::kDynamic) return {-1, -1};
      
      int64_t stride = 1; 
      if (i < strides.size() && strides[i] != ShapedType::kDynamic) {
          stride = strides[i];
      } else {
          return {-1, -1};
      }
      
      totalOffset += off * stride;
    }
 
    return {totalOffset * elemSize, newSize};
  }
 
  // === Case 2: memref.reinterpret_cast ===
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    auto staticOffsets = castOp.getStaticOffsets();
    if (staticOffsets.empty() || staticOffsets[0] == ShapedType::kDynamic) {
        return {0, 0};
    }
    return {staticOffsets[0] * elemSize, 0}; 
  }
 
  return {0, 0};
}
 
// ============================================================================
// 1. 构建入口
// ============================================================================
void PTOIRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  RecursionIR(&funcRegion);
  FinalizeExplicitSubviewMultibufferGroups();
}
 
// ============================================================================
// 2. 更新 Kernel 参数内存信息 (GM Global Memory)
// ============================================================================
void PTOIRTranslator::UpdateKernelArgMemInfo() {
  auto funcParamSize = func_.getNumArguments();
  for (size_t i = 0; i < funcParamSize; i++) {
    Value funcArg = func_.getArgument(i);
    Type argType = funcArg.getType();
 
    if (!isa<pto::PtrType>(argType) && !isa<MemRefType>(argType)) {
      continue;
    }
 
    std::unique_ptr<BaseMemInfo> newMemInfo = std::make_unique<BaseMemInfo>(
        funcArg,                  // baseBuffer
        funcArg,                  // rootBuffer
        pto::AddressSpace::GM,    // Scope
        SmallVector<uint64_t>{0}, // Base Addresses
        0                         // Allocate Size
    );
 
    buffer2MemInfoMap_[funcArg].emplace_back(newMemInfo->clone());
  }
}
 
// ============================================================================
// 3. 递归遍历 IR (核心分发逻辑)
// ============================================================================
void PTOIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {

    // --- Case A: 内存分配 (AllocTile) ---
    if (auto allocOp = dyn_cast<pto::AllocTileOp>(op)) {
      if (failed(UpdateAllocTileOpMemInfo(allocOp))) {
        return WalkResult::interrupt();
      }
    }
    // 支持标准 memref.alloc
    else if (auto memAllocOp = dyn_cast<memref::AllocOp>(op)) {
       if (failed(UpdateMemrefAllocOpMemInfo(memAllocOp))) {
          return WalkResult::interrupt();
       }
    }
    else if (auto declareOp = dyn_cast<pto::DeclareTileMemRefOp>(op)) {
      if (failed(UpdateDeclareTileMemRefOpMemInfo(declareOp))) {
        return WalkResult::interrupt();
      }
    }
    else if (auto castOp = dyn_cast<pto::PointerCastOp>(op)) {
      if (failed(UpdatePointerCastOpMemInfo(castOp))) return WalkResult::interrupt();
    }
    
    // --- Case B: 别名/视图操作 ---
    else if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
      UpdateAliasBufferInfo(makeViewOp.getResult(), makeViewOp.getPtr());
    } 
    else if (auto bindTileOp = dyn_cast<pto::BindTileOp>(op)) {
      UpdateAliasBufferInfo(bindTileOp.getResult(), bindTileOp.getSource());
    }
    else if (auto subViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
      UpdateAliasBufferInfo(subViewOp.getResult(), subViewOp.getSource());
    } 
    else if (auto memrefSubView = dyn_cast<memref::SubViewOp>(op)) {
      UpdateAliasBufferInfo(memrefSubView.getResult(), memrefSubView.getSource());
    }
    else if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
      UpdateAliasBufferInfo(castOp.getResult(), castOp.getSource());
    }
    // [Fix] 添加 CollapseShape 和 ExpandShape 的支持
    else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
      UpdateAliasBufferInfo(collapseOp.getResult(), collapseOp.getSrc());
    }
    else if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
      UpdateAliasBufferInfo(expandOp.getResult(), expandOp.getSrc());
    }
 
    // --- Case C: 控制流 (SCF) ---
    else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      UpdateForOpInfo(forOp);
      return WalkResult::skip();
    } 
    else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      UpdateWhileOpInfo(whileOp);
      return WalkResult::skip();
    } 
    else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      UpdateIfOpInfo(ifOp);
      return WalkResult::skip();
    } 
    else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      UpdateYieldOpInfo(yieldOp);
    }
    // --- Case D: 带有 OpPipeInterface 的计算/搬运指令 ---
    else if (isa<pto::OpPipeInterface>(op)) {
      UpdatePTOOpInfo(op);
    }
    
    return WalkResult::advance();
  });
 
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PTO InjectSync Traverse IR Failed!");
  }
}
 
// ============================================================================
// 4. 处理 AllocTile / PointerCast
// ============================================================================
LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
  Value res = op.getResult();
  
  auto tileType = dyn_cast<pto::TileBufType>(res.getType());
  uint64_t sizeInBytes = 0;
  uint64_t baseAddr = 0;

  // If alloc_tile carries an explicit address, record it when it's a constant.
  if (Value addr = op.getAddr()) {
    llvm::APInt apIntValue;
    if (matchPattern(addr, m_ConstantInt(&apIntValue))) {
        // 将 APInt 转换为 int64_t，再转为 uint64_t
        int64_t c = apIntValue.getSExtValue();  // 有符号扩展转换
        // 如果确定是无符号值，也可以用：apIntValue.getZExtValue()
        baseAddr = static_cast<uint64_t>(c);
    }
  }

  // 1. 计算大小
  if (tileType) {
    ArrayRef<int64_t> shape = tileType.getShape();
    bool isStatic = true;
    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic) {
        isStatic = false;
        break;
      }
    }

    if (isStatic) {
      int64_t elemSize = tileType.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t numElements = 1;
      for (auto dim : shape) numElements *= dim;
      sizeInBytes = numElements * elemSize;
    }
  }

  // 2. 解析地址空间
  // 默认设为 MAT (Matrix Buffer)，但优先读取 Type 中的属性
  pto::AddressSpace space = pto::AddressSpace::MAT; 
  
  if (tileType) {
      if (auto attr = tileType.getMemorySpace()) {
          // 尝试转换为 PTO 的 AddressSpaceAttr
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
              space = ptoAttr.getAddressSpace();
          }
      }
  }

  // 3. 注册 Buffer 信息
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,                  
      res,                  
      space, // 使用解析出的 space                 
      SmallVector<uint64_t>{baseAddr},
      sizeInBytes             
  );

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
LogicalResult PTOIRTranslator::UpdatePointerCastOpMemInfo(pto::PointerCastOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();
 
  if (op.getAddrs().empty()) {
    return op.emitError("PointerCast must have at least one address operand");
  }
  SmallVector<uint64_t> baseAddresses;
  baseAddresses.reserve(op.getAddrs().size());
  for (Value addr : op.getAddrs()) {
    llvm::APInt apIntValue;
    if (!matchPattern(addr, m_ConstantInt(&apIntValue))) {
      // Variable address: be conservative and treat as unknown overlap.
      baseAddresses.clear();
      break;
    }
    int64_t c = apIntValue.getSExtValue();
    if (c < 0) {
      // Unexpected negative planned address: drop address info to stay
      // conservative in dependency analysis.
      baseAddresses.clear();
      break;
    }
    baseAddresses.push_back(static_cast<uint64_t>(c));
  }

  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t bitWidth =
        memRefType.getElementType().getIntOrFloatBitWidth();
    uint64_t elemBytes = static_cast<uint64_t>((bitWidth + 7) / 8);
    if (elemBytes == 0)
      elemBytes = 1;

    // Prefer stride-based size computation to account for padded/fractal layouts.
    SmallVector<int64_t> strides;
    int64_t offset = ShapedType::kDynamic;
    if (succeeded(getStridesAndOffset(memRefType, strides, offset)) &&
        offset != ShapedType::kDynamic &&
        llvm::all_of(strides, [](int64_t s) { return s != ShapedType::kDynamic; }) &&
        offset >= 0) {
      uint64_t maxIndex = static_cast<uint64_t>(offset);
      auto shape = memRefType.getShape();
      bool invalid = false;
      for (size_t i = 0; i < shape.size(); ++i) {
        int64_t dim = shape[i];
        if (dim <= 0) {
          invalid = true;
          break;
        }
        uint64_t stride = static_cast<uint64_t>(strides[i]);
        maxIndex += static_cast<uint64_t>(dim - 1) * stride;
      }
      if (!invalid && !shape.empty()) {
        sizeInBytes = (maxIndex + 1) * elemBytes;
      }
    } else {
      uint64_t numElements = 1;
      for (auto dim : memRefType.getShape())
        numElements *= static_cast<uint64_t>(dim);
      sizeInBytes = numElements * elemBytes;
    }
  }
 
  pto::AddressSpace space = pto::AddressSpace::GM; 
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }
 
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,          
      res,
      space,
      std::move(baseAddresses),
      sizeInBytes
  );
 
  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareTileMemRefOpMemInfo(pto::DeclareTileMemRefOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType)
    return failure();

  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elemSize == 0)
      elemSize = 1;

    int64_t numElements = 1;
    for (auto dim : memRefType.getShape())
      numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr))
      space = ptoAttr.getAddressSpace();
  }

  // declare_tile_memref is only a symbolic placeholder. Use its SSA result as
  // both base/root so later bind_tile aliases and tpop consumers can be
  // connected by InsertSync without inventing a fake allocation.
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      space,
      SmallVector<uint64_t>{0},
      sizeInBytes);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
// ============================================================================
// 5. [P0 修改] 更新 PTO Op 信息 (通用接口版)
// ============================================================================
void PTOIRTranslator::UpdatePTOOpInfo(Operation *op) {
  // 1. 获取流水线类型 (现在通过 Interface)
  pto::PipelineType pipe = getOpPipeline(op);
  
  // 如果 Op 不属于任何关心的流水线，直接跳过，不建立 Sync 节点
  if (pipe == pto::PipelineType::PIPE_UNASSIGNED) return;
 
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
 
  // 2. [关键] 使用 MemoryEffects 接口自动获取读写依赖
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
     SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
     memEffect.getEffects(effects);
     
     for (auto &effect : effects) {
       Value val = effect.getValue();
       if (!val) continue;
 
       // 只有当 Value 在我们的 BufferMap 中有记录时，才视为有效依赖
       // (过滤掉比如 Loop Iterator 或其他标量)
       if (isa<MemoryEffects::Read>(effect.getEffect())) {
          UpdateDefUseVec({val}, useVec);
       } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          UpdateDefUseVec({val}, defVec);
       }
     }
  } else {
    // 如果算子有 Pipe 属性但没实现 MemoryEffects，这是一个定义错误
    // 我们可以打印个 Warning 或者保持为空 (认为无副作用)
    LLVM_DEBUG(llvm::dbgs() << "Warning: Op " << op->getName() 
                            << " has Pipe but no MemoryEffects interface.\n");
  }
 
  // 3. 构建 Compound Node
  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, op->getName());
  compoundElement->elementOp = op;
 
  // 4. 设置 Core Type (用于区分 Cube/Vector 资源)
  // Matmul (M) 和 L1->L0 搬运 (MTE1) 通常涉及 Cube 资源
  if (pipe == pto::PipelineType::PIPE_M || pipe == pto::PipelineType::PIPE_MTE1) {
    compoundElement->compoundCoreType = pto::TCoreType::CUBE; 
  } else {
    // MTE2, MTE3, Vector 归类为 Vector Core (或者对应 MTE 资源)
    compoundElement->compoundCoreType = pto::TCoreType::VECTOR;
  }
 
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}
 
// ============================================================================
// 6. [P0 修改] 获取 Op 的 Pipeline 类型
// ============================================================================
pto::PipelineType PTOIRTranslator::getOpPipeline(Operation *op) {
  // 1. 优先尝试通过接口获取
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
    // 注意：假设 pto::Pipe (ODS Enum) 和 pto::PipelineType (C++ Enum) 的数值定义是一致的
    // 或者在这里做一个 switch-case 映射
    // 目前假设直接 cast 是安全的 (0=S, 1=V, 2=M ...)
    return static_cast<pto::PipelineType>(pipeOp.getPipe());
  }
 
  // 2. 如果没实现接口，返回 Unassigned
  return pto::PipelineType::PIPE_UNASSIGNED;
}
 
// ============================================================================
// 7. 控制流处理 (SCF Support)
// ============================================================================
 
void PTOIRTranslator::UpdateForOpInfo(scf::ForOp forOp) {
  auto forBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  forBeginElement->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forBeginElement));
  
  std::unique_ptr<InstanceElement> &forElement = syncIR_[index];
  index++;
  
  auto *forBeginPtr = dyn_cast<LoopInstanceElement>(forElement.get());
  assert(forBeginPtr != nullptr && "Sync IR Construction failed.");
  
  if (!forOp.getInitArgs().empty()) {
    assert(forOp.getInitArgs().size() == forOp.getRegionIterArgs().size());
    for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
      UpdateAliasBufferInfo(forOp.getRegionIterArgs()[i], arg);
    }
  }
 
  RecursionIR(&forOp.getRegion());
 
  forBeginPtr->endId = index;
  auto forEnd = forBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}
 
void PTOIRTranslator::UpdateWhileOpInfo(scf::WhileOp whileOp) {
  auto loopBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  loopBeginElement->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(loopBeginElement));
  
  auto *loopBeginPtr = dyn_cast<LoopInstanceElement>(syncIR_.back().get());
  index++;
 
  if (!whileOp.getInits().empty()) {
    for (auto [initArg, blockArg] : llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
      UpdateAliasBufferInfo(blockArg, initArg);
    }
    auto conditionOp = whileOp.getConditionOp();
    for (auto [yieldedArg, blockArg] : llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
      UpdateAliasBufferInfo(blockArg, yieldedArg);
    }
  }
 
  RecursionIR(&whileOp.getBefore());
  RecursionIR(&whileOp.getAfter());
 
  loopBeginPtr->endId = index;
  auto forEnd = loopBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}
 
void PTOIRTranslator::UpdateIfOpInfo(scf::IfOp ifOp) {
  auto ifBeginElement = std::make_unique<BranchInstanceElement>(index, index, KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  auto *ifPtr = ifBeginElement.get();
  
  syncIR_.emplace_back(std::move(ifBeginElement));
  index++;
 
  // 1. 处理 Then 区域
  RecursionIR(&ifOp.getThenRegion());
  
  // Then 的结束占位符
  auto placeHolder = std::make_unique<PlaceHolderInstanceElement>(index, ifPtr->GetIndex());

  // 直接指向Then Block的yieldop
  placeHolder->elementOp = ifOp.getThenRegion().front().getTerminator();

  syncIR_.emplace_back(std::move(placeHolder));
  index++;
  
  ifPtr->branchId = index;
 
  // 2. 处理 Else 区域 (总是创建 SyncIR 节点，即使 IR 中没有 Else)
  auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
  ifElseElement->elementOp = ifOp.getOperation();
  auto *elsePtr = ifElseElement.get();
 
  syncIR_.emplace_back(std::move(ifElseElement));
  index++;
 
  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion());
  }
  
  // Else 的结束占位符
  auto elsePlaceHolder = std::make_unique<PlaceHolderInstanceElement>(index, elsePtr->GetIndex());
  
  if (ifOp.elseBlock()) {
      // 如果有真实的 Else Block，映射到 ifOp (CodeGen 需定位到 Else Yield 前)
      elsePlaceHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      elsePlaceHolder->isVirtualElse = false;
  } else {
      // 如果没有 Else Block，标记为虚拟，映射到 ifOp
      elsePlaceHolder->elementOp = ifOp.getOperation();
      elsePlaceHolder->isVirtualElse = true;
      elsePlaceHolder->parentIfOp = ifOp.getOperation();
  }
  
  syncIR_.emplace_back(std::move(elsePlaceHolder));
  index++;
  
  elsePtr->endId = index;
  ifPtr->endId = index;
 
  // 3. If End
  auto ifEndElement = ifPtr->CloneBranch(KindOfBranch::IF_END);
  ifEndElement->elementOp = ifOp.getOperation();
  syncIR_.emplace_back(std::move(ifEndElement));
  index++;
}
 
void PTOIRTranslator::UpdateYieldOpInfo(scf::YieldOp yieldOp) {
  auto *parentOp = yieldOp->getParentOp();
  if (!parentOp || isa<scf::WhileOp>(parentOp)) return;
 
  assert(parentOp->getResults().size() == yieldOp->getOpOperands().size());
  for (auto [yieldVal, resultVal] : llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
}
 
// ============================================================================
// 8. 辅助函数
// ============================================================================
void PTOIRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;
 
  int64_t deltaOffset = 0;
  int64_t newSize = -1; 
 
  if (auto op = result.getDefiningOp()) {
    auto info = getStaticOffsetAndSize(op, source);
    if (info.first != -1) {
        deltaOffset = info.first;
        if (info.second > 0) newSize = info.second;
    } 
  }
 
  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
 
    if (!newInfo->baseAddresses.empty()) {
      if (deltaOffset < 0) {
        // Negative offsets are unexpected for buffer views in this pipeline.
        // Drop address information to stay conservative in dependency analysis.
        newInfo->baseAddresses.clear();
      } else {
        for (auto &addr : newInfo->baseAddresses)
          addr += static_cast<uint64_t>(deltaOffset);
      }
    }
 
    if (newSize > 0) {
        newInfo->allocateSize = newSize;
    }

    TryMarkSubviewMultibufferSlot(result, source, *parentInfo, *newInfo);

    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}

void PTOIRTranslator::TryMarkSubviewMultibufferSlot(
    Value result, Value source, const BaseMemInfo &parentInfo,
    BaseMemInfo &newInfo) {
  Value multibufferRoot;
  int multibufferSlot = -1;
  int multibufferFactor = 1;
  int multibufferGroup = 0;
  if (IsSubviewMultibufferRootInvalid(parentInfo.rootBuffer) ||
      IsSubviewMultibufferRootInvalid(newInfo.rootBuffer)) {
    newInfo.multibufferRoot = nullptr;
    newInfo.multibufferSlot = -1;
    newInfo.multibufferFactor = 1;
    newInfo.multibufferGroup = 0;
    newInfo.isMultibufferSlotValid = false;
    newInfo.suppressLegacyMultibuffer = true;
    return;
  }

  if (!TryComputeSubviewSlotInfo(result.getDefiningOp(), source, newInfo,
                                 multibufferRoot, multibufferSlot,
                                 multibufferFactor, multibufferGroup)) {
    if (isSubviewLikeOp(result.getDefiningOp()) &&
        (HasExplicitSubviewMultibufferAnnotation(result.getDefiningOp()) ||
         IsRootLevelSubviewMultibufferCandidate(parentInfo))) {
      InvalidateSubviewMultibufferRoot(parentInfo.rootBuffer);
      newInfo.suppressLegacyMultibuffer = true;
    }
    return;
  }

  if (IsSubviewMultibufferRootInvalid(multibufferRoot)) {
    newInfo.multibufferRoot = nullptr;
    newInfo.multibufferSlot = -1;
    newInfo.multibufferFactor = 1;
    newInfo.multibufferGroup = 0;
    newInfo.isMultibufferSlotValid = false;
    newInfo.suppressLegacyMultibuffer = true;
    return;
  }

  newInfo.multibufferRoot = multibufferRoot;
  newInfo.multibufferSlot = multibufferSlot;
  newInfo.multibufferFactor = multibufferFactor;
  newInfo.multibufferGroup = multibufferGroup;
  newInfo.isMultibufferSlotValid = true;
  newInfo.suppressLegacyMultibuffer = false;
}

bool PTOIRTranslator::TryComputeSubviewSlotInfo(Operation *op, Value source,
                                                const BaseMemInfo &parentInfo,
                                                Value &multibufferRoot,
                                                int &multibufferSlot,
                                                int &multibufferFactor,
                                                int &multibufferGroup) {
  (void)source;
  if (!op) return false;

  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> sourceShape;

  auto fillForPTOSubview = [&](pto::SubViewOp subviewOp) -> bool {
    auto rootType = dyn_cast<pto::TileBufType>(subviewOp.getSource().getType());
    if (!rootType) return false;
    if (subviewOp.getOffsets().size() != subviewOp.getSizes().size()) return false;
    if (rootType.getShape().size() != subviewOp.getSizes().size()) return false;

    sourceShape.assign(rootType.getShape().begin(), rootType.getShape().end());
    offsets.reserve(subviewOp.getOffsets().size());
    sizes.reserve(subviewOp.getSizes().size());
    for (Value off : subviewOp.getOffsets()) {
      auto c = getConstantIntValue(off);
      if (!c || *c < 0) return false;
      offsets.push_back(*c);
    }
    for (Attribute sizeAttr : subviewOp.getSizes()) {
      int64_t size = cast<IntegerAttr>(sizeAttr).getInt();
      if (size == ShapedType::kDynamic || size <= 0) return false;
      sizes.push_back(size);
    }
    return true;
  };

  auto fillForSubView = [&](memref::SubViewOp subView) -> bool {
    auto memrefType = dyn_cast<MemRefType>(subView.getSource().getType());
    if (!memrefType || !memrefType.hasStaticShape()) return false;
    sourceShape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    offsets.reserve(subView.getStaticOffsets().size());
    sizes.reserve(subView.getStaticSizes().size());
    for (int64_t off : subView.getStaticOffsets()) {
      if (off == ShapedType::kDynamic || off < 0) return false;
      offsets.push_back(off);
    }
    for (int64_t size : subView.getStaticSizes()) {
      if (size == ShapedType::kDynamic || size <= 0) return false;
      sizes.push_back(size);
    }
    for (int64_t stride : subView.getStaticStrides()) {
      if (stride == ShapedType::kDynamic || stride <= 0) return false;
    }
    return true;
  };

  bool matched = false;
  if (auto subviewOp = dyn_cast<pto::SubViewOp>(op)) {
    matched = fillForPTOSubview(subviewOp);
  } else if (auto subView = dyn_cast<memref::SubViewOp>(op)) {
    matched = fillForSubView(subView);
  }
  if (!matched) return false;

  int annotatedSlot = -1;
  int annotatedFactor = 1;
  int annotatedGroup = 0;
  bool hasExplicitAnnotation = TryGetAnnotatedSubviewMultibufferInfo(
      op, annotatedSlot, annotatedFactor, annotatedGroup);

  if (!hasExplicitAnnotation && !IsRootMarkedAsPingpong(parentInfo.rootBuffer) &&
      !isLegacyDoubleBufferRoot(parentInfo))
    return false;
  if (offsets.size() != sizes.size() || offsets.size() != sourceShape.size())
    return false;

  int factor = hasExplicitAnnotation ? annotatedFactor : 2;
  if (factor <= 1) return false;

  for (size_t i = 0; i < offsets.size(); ++i) {
    int64_t rootDim = sourceShape[i];
    int64_t offset = offsets[i];
    int64_t size = sizes[i];
    if (rootDim == ShapedType::kDynamic || rootDim <= 0) return false;
    if (offset + size > rootDim) return false;
  }

  if (hasExplicitAnnotation) {
    explicitSubviewMultibufferCandidates_.push_back(
        ExplicitSubviewMultibufferCandidate{
            op->getResult(0),
            parentInfo.rootBuffer,
            annotatedGroup,
            annotatedSlot,
            factor,
            SmallVector<int64_t, 4>(offsets.begin(), offsets.end()),
            SmallVector<int64_t, 4>(sizes.begin(), sizes.end()),
            SmallVector<int64_t, 4>(sourceShape.begin(), sourceShape.end())});
    multibufferRoot = parentInfo.rootBuffer;
    multibufferSlot = annotatedSlot;
    multibufferFactor = factor;
    multibufferGroup = annotatedGroup;
    return true;
  }

  int partitionDim = -1;
  int64_t partitionExtent = 0;
  for (size_t i = 0; i < offsets.size(); ++i) {
    int64_t rootDim = sourceShape[i];
    int64_t offset = offsets[i];
    int64_t size = sizes[i];
    if (size == rootDim) {
      if (offset != 0) return false;
      continue;
    }

    if (partitionDim != -1) return false;
    partitionDim = static_cast<int>(i);
    partitionExtent = rootDim;
  }

  if (partitionDim < 0 || partitionExtent <= 0) return false;
  if (partitionExtent % factor != 0) return false;
  if (sizes[static_cast<size_t>(partitionDim)] * factor != partitionExtent)
    return false;
  if (offsets[static_cast<size_t>(partitionDim)] %
          sizes[static_cast<size_t>(partitionDim)] !=
      0)
    return false;

  int64_t slot = offsets[static_cast<size_t>(partitionDim)] /
                 sizes[static_cast<size_t>(partitionDim)];
  if (slot < 0 || slot >= factor) return false;

  auto rootAllocBytes = getStaticValueSizeInBytes(parentInfo.rootBuffer);
  auto slotAllocBytes = getStaticValueSizeInBytes(op->getResult(0));
  if (rootAllocBytes && slotAllocBytes && *rootAllocBytes > 0 &&
      *slotAllocBytes > 0 &&
      static_cast<uint64_t>(*rootAllocBytes) !=
          static_cast<uint64_t>(*slotAllocBytes) *
              static_cast<uint64_t>(factor))
    return false;

  multibufferRoot = parentInfo.rootBuffer;
  multibufferSlot = static_cast<int>(slot);
  multibufferFactor = factor;
  multibufferGroup = 0;
  return true;
}

bool PTOIRTranslator::TryGetAnnotatedSubviewMultibufferInfo(
    Operation *op, int &multibufferSlot, int &multibufferFactor,
    int &multibufferGroup) const {
  if (!op) return false;

  auto factorAttr =
      op->getAttrOfType<IntegerAttr>(kSubviewMultiBufferFactorAttr);
  auto slotAttr = op->getAttrOfType<IntegerAttr>(kSubviewMultiBufferSlotAttr);
  auto groupAttr = op->getAttrOfType<IntegerAttr>(kSubviewMultiBufferGroupAttr);

  if (!factorAttr && !slotAttr && !groupAttr)
    return false;
  if (!factorAttr || !slotAttr)
    return false;
  if (groupAttr && groupAttr.getInt() < 0)
    return false;

  multibufferFactor = static_cast<int>(factorAttr.getInt());
  multibufferSlot = static_cast<int>(slotAttr.getInt());
  multibufferGroup = groupAttr ? static_cast<int>(groupAttr.getInt()) : 0;
  if (!isSupportedExplicitSubviewMultibufferFactor(multibufferFactor))
    return false;
  if (multibufferSlot < 0 || multibufferSlot >= multibufferFactor)
    return false;
  return true;
}

bool PTOIRTranslator::HasExplicitSubviewMultibufferAnnotation(Operation *op) const {
  if (!op) return false;
  return op->hasAttr(kSubviewMultiBufferFactorAttr) ||
         op->hasAttr(kSubviewMultiBufferSlotAttr) ||
         op->hasAttr(kSubviewMultiBufferGroupAttr);
}

bool PTOIRTranslator::IsRootMarkedAsPingpong(Value root) const {
  if (!root) return false;
  if (Operation *defOp = root.getDefiningOp()) {
    auto attr = defOp->getAttrOfType<IntegerAttr>(kLegacyMultiBufferAttr);
    return attr && attr.getInt() == 2;
  }
  return false;
}

bool PTOIRTranslator::IsSubviewMultibufferRootInvalid(Value root) const {
  return root && invalidSubviewMultibufferRoots_.contains(root);
}

bool PTOIRTranslator::IsRootLevelSubviewMultibufferCandidate(
    const BaseMemInfo &parentInfo) const {
  if (!parentInfo.rootBuffer) return false;
  if (!IsRootMarkedAsPingpong(parentInfo.rootBuffer) &&
      !isLegacyDoubleBufferRoot(parentInfo))
    return false;
  return parentInfo.baseBuffer == parentInfo.rootBuffer ||
         parentInfo.allocateSize > 0;
}

void PTOIRTranslator::InvalidateSubviewMultibufferRoot(Value root) {
  if (!root) return;
  invalidSubviewMultibufferRoots_.insert(root);
  for (auto &entry : buffer2MemInfoMap_) {
    for (auto &info : entry.second) {
      if (!info || info->rootBuffer != root) continue;
      info->multibufferRoot = nullptr;
      info->multibufferSlot = -1;
      info->multibufferFactor = 1;
      info->multibufferGroup = 0;
      info->isMultibufferSlotValid = false;
      info->suppressLegacyMultibuffer = true;
    }
  }
}

void PTOIRTranslator::FinalizeExplicitSubviewMultibufferGroups() {
  llvm::DenseMap<Value, SmallVector<const ExplicitSubviewMultibufferCandidate *, 8>>
      candidatesByRoot;
  for (const auto &candidate : explicitSubviewMultibufferCandidates_) {
    if (!candidate.rootBuffer)
      continue;
    candidatesByRoot[candidate.rootBuffer].push_back(&candidate);
  }

  for (const auto &[root, candidates] : candidatesByRoot) {
    if (IsSubviewMultibufferRootInvalid(root))
      continue;
    if (!ValidateExplicitSubviewMultibufferRoot(root, candidates))
      InvalidateSubviewMultibufferRoot(root);
  }
}

bool PTOIRTranslator::ValidateExplicitSubviewMultibufferRoot(
    Value root,
    ArrayRef<const ExplicitSubviewMultibufferCandidate *> candidates) {
  if (!root || candidates.empty())
    return false;

  struct GroupRegion {
    SmallVector<int64_t, 4> offsets;
    SmallVector<int64_t, 4> sizes;
  };

  llvm::DenseMap<int, SmallVector<const ExplicitSubviewMultibufferCandidate *, 8>>
      candidatesByGroup;
  for (const auto *candidate : candidates) {
    if (!candidate || candidate->rootBuffer != root)
      return false;
    candidatesByGroup[candidate->multibufferGroup].push_back(candidate);
  }

  SmallVector<GroupRegion, 4> groupRegions;
  for (const auto &[group, groupCandidates] : candidatesByGroup) {
    (void)group;
    if (groupCandidates.empty())
      return false;

    const auto *first = groupCandidates.front();
    const size_t rank = first->offsets.size();
    const int factor = first->multibufferFactor;
    if (rank == 0 || rank != first->sizes.size() ||
        rank != first->sourceShape.size() || factor <= 1)
      return false;

    for (const auto *candidate : groupCandidates) {
      if (!candidate || candidate->offsets.size() != rank ||
          candidate->sizes.size() != rank ||
          candidate->sourceShape.size() != rank)
        return false;
      if (candidate->multibufferFactor != factor)
        return false;
      if (candidate->multibufferSlot < 0 || candidate->multibufferSlot >= factor)
        return false;
      if (candidate->sizes != first->sizes ||
          candidate->sourceShape != first->sourceShape)
        return false;
    }

    SmallVector<int64_t, 4> chosenBaseOffsets;
    SmallVector<int64_t, 4> chosenGroupSizes;
    bool foundSlotDim = false;
    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t slotBase =
          first->offsets[dim] -
          static_cast<int64_t>(first->multibufferSlot) * first->sizes[dim];
      if (slotBase < 0)
        continue;
      if (slotBase + static_cast<int64_t>(factor) * first->sizes[dim] >
          first->sourceShape[dim])
        continue;

      bool validSlotDim = true;
      SmallVector<int64_t, 4> baseOffsets(first->offsets.begin(),
                                          first->offsets.end());
      baseOffsets[dim] = slotBase;
      for (const auto *candidate : groupCandidates) {
        if (candidate->offsets[dim] -
                static_cast<int64_t>(candidate->multibufferSlot) *
                    candidate->sizes[dim] !=
            slotBase) {
          validSlotDim = false;
          break;
        }
        for (size_t otherDim = 0; otherDim < rank; ++otherDim) {
          if (otherDim == dim)
            continue;
          if (candidate->offsets[otherDim] != first->offsets[otherDim]) {
            validSlotDim = false;
            break;
          }
        }
        if (!validSlotDim)
          break;
      }
      if (!validSlotDim)
        continue;

      SmallVector<int64_t, 4> groupSizes(first->sizes.begin(),
                                         first->sizes.end());
      groupSizes[dim] *= factor;
      chosenBaseOffsets = std::move(baseOffsets);
      chosenGroupSizes = std::move(groupSizes);
      foundSlotDim = true;
      break;
    }

    if (!foundSlotDim)
      return false;
    groupRegions.push_back({std::move(chosenBaseOffsets),
                            std::move(chosenGroupSizes)});
  }

  for (size_t i = 0; i < groupRegions.size(); ++i) {
    for (size_t j = i + 1; j < groupRegions.size(); ++j) {
      if (areStaticBoxesOverlapping(groupRegions[i].offsets, groupRegions[i].sizes,
                                    groupRegions[j].offsets,
                                    groupRegions[j].sizes)) {
        return false;
      }
    }
  }
  return true;
}
 
// ============================================================================
// 实现 UpdateMemrefAllocOpMemInfo
// ============================================================================
LogicalResult PTOIRTranslator::UpdateMemrefAllocOpMemInfo(memref::AllocOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();
 
  // 1. 计算大小 (Bytes)
  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elemSize == 0) elemSize = 1; // bool case
    
    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }
 
  // 2. 解析地址空间 (Scope)
  // 默认视为 MAT/UB (Local Memory)，这是 alloc 的常见用途
  // 如果有显式属性，则覆盖
  pto::AddressSpace space = pto::AddressSpace::MAT; 
  
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }
 
  // 3. 注册 Buffer 信息
  // 对于 alloc，它自己就是 Root
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,                      // baseBuffer
      res,                      // rootBuffer (Self is root)
      space,
      SmallVector<uint64_t>{0}, // Base Addresses (Offset 0)
      sizeInBytes
  );
 
  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
void PTOIRTranslator::UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec) {
  for (Value v : values) {
    if (buffer2MemInfoMap_.contains(v)) {
      for (auto &memInfo : buffer2MemInfoMap_[v]) {
        vec.push_back(memInfo.get());
      }
    }
  }
}
 
// ============================================================================
// 9. 调试与打印支持
// ============================================================================
 
std::string PTOIRTranslator::getPipelineName(pto::PipelineType pipe) {
  switch (pipe) {
  case pto::PipelineType::PIPE_MTE1: return "MTE1";
  case pto::PipelineType::PIPE_MTE2: return "MTE2";
  case pto::PipelineType::PIPE_MTE3: return "MTE3";
  case pto::PipelineType::PIPE_M:    return "CUBE";
  case pto::PipelineType::PIPE_V:    return "VECTOR";
  case pto::PipelineType::PIPE_S:    return "SCALAR";
  case pto::PipelineType::PIPE_ALL:  return "BARRIER";
  default: return "UNKNOWN";
  }
}
 
void PTOIRTranslator::printMemInfoList(llvm::raw_ostream &os, 
                                       const SmallVector<const BaseMemInfo *> &list, 
                                       AsmState &state) {
  os << "[";
  bool first = true;
  for (const auto *info : list) {
    if (!first) os << ", ";
    info->rootBuffer.printAsOperand(os, state);
    // [Fix] 打印 MAT 或 VEC 或 GM
    if (info->scope == pto::AddressSpace::GM) os << "(GM)";
    else if (info->scope == pto::AddressSpace::MAT) os << "(MAT)";
    else if (info->scope == pto::AddressSpace::VEC) os << "(VEC)";
    else os << "(Other)"; // 处理 LEFT/RIGHT/ACC 等其他情况
    first = false;
  }
  os << "]";
}
 
void PTOIRTranslator::print() {
  llvm::errs() << "\n=== PTO IR Translator Dump ===\n";
  
  AsmState state(func_); 
 
  llvm::errs() << "--- Buffer Analysis (Value -> Root) ---\n";
  for (auto &it : buffer2MemInfoMap_) {
    Value v = it.first;
    auto &infoList = it.second;
    
    llvm::errs() << "  ";
    v.printAsOperand(llvm::errs(), state);
    llvm::errs() << " -> ";
    
    for (auto &mem : infoList) {
        mem->rootBuffer.printAsOperand(llvm::errs(), state);
        llvm::errs() << " ";
    }
    llvm::errs() << "\n";
  }
 
  llvm::errs() << "\n--- SyncIR Structure ---\n";
  for (const auto &element : syncIR_) {
    unsigned id = element->GetIndex();
    llvm::errs() << llvm::formatv("{0,4}: ", id); 
 
    switch (element->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = dyn_cast<CompoundInstanceElement>(element.get());
      llvm::errs() << "COMPOUND [" << getPipelineName(comp->kPipeValue) << "] ";
      llvm::errs() << comp->opName.getStringRef() << "\n";
      
      llvm::errs() << "      DEF: ";
      printMemInfoList(llvm::errs(), comp->defVec, state);
      llvm::errs() << "\n      USE: ";
      printMemInfoList(llvm::errs(), comp->useVec, state);
      llvm::errs() << "\n";
      break;
    }
    case InstanceElement::KindTy::LOOP: 
        llvm::errs() << "LOOP\n"; break;
    case InstanceElement::KindTy::BRANCH: 
        llvm::errs() << "BRANCH\n"; break;
    case InstanceElement::KindTy::PLACE_HOLDER: 
        llvm::errs() << "PLACE_HOLDER\n"; break;
    }
  }
  llvm::errs() << "==============================\n\n";
}
