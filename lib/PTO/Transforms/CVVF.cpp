// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "PTO/Transforms/CVVF.h"
#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOAddressSemantics.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VmiMemoryLocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;
using namespace mlir::pto;

namespace {
Value traceCVStorageRoot(Value value, bool followCarried) {
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (auto *op = value.getDefiningOp()) {
    if (!seen.insert(op).second) return {};
    if (auto addr = dyn_cast<TileBufAddrOp>(op)) value = addr.getSrc();
    else if (auto add = dyn_cast<AddPtrOp>(op)) value = add.getPtr();
    else if (auto reshape = dyn_cast<TReshapeOp>(op)) value = reshape.getSrc();
    else if (auto view = dyn_cast<SubViewOp>(op)) value = view.getSource();
    else if (auto cast = dyn_cast<BitcastOp>(op)) value = cast.getSrc();
    else if (auto cast = dyn_cast<CastPtrOp>(op)) value = cast.getInput();
    else if (auto alloc = dyn_cast<AllocTileOp>(op)) {
      // A helper's alloc_tile is a view of a pointer argument, not storage.
      if (alloc->hasAttr("pto.cv.borrowed")) value = alloc.getAddr();
      else return value;
    } else {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        auto result = dyn_cast<OpResult>(value);
        if (result && result.getResultNumber() < loop.getInitArgs().size()) {
          unsigned index = result.getResultNumber();
          auto yield = mlir::cast<scf::YieldOp>(loop.getBody()->getTerminator());
          if (traceCVStorageRoot(yield.getOperand(index), false) == loop.getRegionIterArgs()[index]) {
            value = loop.getInitArgs()[index]; continue;
          }
          return {};
        }
      }
      auto address = getDefaultVPTOAddressSemantics(op);
      if (address.postUpdate && address.postUpdate->updatedBase == value) {
        value = address.postUpdate->baseOperand->get();
        continue;
      }
      return value;
    }
  }
  if (followCarried) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())) {
        unsigned index = argument.getArgNumber();
        if (index && index <= loop.getInitArgs().size()) {
          auto yield = mlir::cast<scf::YieldOp>(loop.getBody()->getTerminator());
          Value incoming = traceCVStorageRoot(yield.getOperand(index - 1), false);
          // Only a post-update chain of this exact carried pointer is proven.
          // A rebind to another storage root remains unknown.
          if (incoming == argument)
            return traceCVStorageRoot(loop.getInitArgs()[index - 1], true);
          return {};
        }
      }
    }
  }
  return value;
}
} // namespace
Value mlir::pto::getCVStorageRoot(Value value) {
  return traceCVStorageRoot(value, true);
}

FailureOr<CVVFEffects> mlir::pto::getCVVFEffects(func::FuncOp function) {
  if (!function || function.isExternal() ||
      !function->hasAttr("pto.cv.prepared_vf")) return failure();
  CVVFEffects effects;
  effects.reads.resize(function.getNumArguments(), false);
  effects.writes.resize(function.getNumArguments(), false);
  bool unknown = false;
  function.walk([&](Operation *op) {
    if (isa<func::FuncOp, func::ReturnOp, AllocTileOp, InitAlignOp>(op) ||
        op->getNumRegions() || isMemoryEffectFree(op)) return;
    auto interface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!interface) {
      op->emitOpError("prepared VF operation has no memory-effect contract");
      unknown = true; return;
    }
    SmallVector<MemoryEffects::EffectInstance> accesses;
    interface.getEffects(accesses);
    for (const auto &access : accesses) {
      if (!isa<MemoryEffects::Read, MemoryEffects::Write>(access.getEffect())) {
        op->emitError("prepared VF operation has an unsupported memory effect");
        unknown = true; continue;
      }
      Value value = access.getValue();
      if (!value) {
        op->emitError("prepared VF memory effect has no bound operand");
        unknown = true; continue;
      }
      // Physical instruction effects also describe register/align state.
      // Only pointer/tile values denote external storage in this contract.
      if (!isa<PtrType, TileBufType>(value.getType())) continue;
      auto argument = dyn_cast<BlockArgument>(getCVStorageRoot(value));
      if (!argument || argument.getOwner() != &function.getBody().front()) {
        op->emitError("prepared VF memory operand does not resolve to an explicit parameter: ") << value;
        unknown = true; continue;
      }
      unsigned i = argument.getArgNumber();
      // VSTUS/VSTUR model pending align state as reads in the generic
      // instruction interface; they nevertheless store to their base. The
      // installed A5 store/flush contract is authoritative for this summary.
      bool alignStore = isa<VstusOp, VsturOp>(op);
      effects.reads[i] = effects.reads[i] || (!alignStore && isa<MemoryEffects::Read>(access.getEffect()));
      effects.writes[i] = effects.writes[i] || alignStore || isa<MemoryEffects::Write>(access.getEffect());
    }
  });
  if (unknown) {
    function.emitError("prepared VF has an unknown memory effect or unbound scratch");
    return failure();
  }
  return effects;
}

FailureOr<CVVFEffects> mlir::pto::getCVVFEffects(func::CallOp call) {
  auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  return getCVVFEffects(function);
}

namespace {
/// Keep the vector computation separate from the outer C/V schedule. All
/// absolute address decisions remain in the caller. The first implementation
/// accepts resultless vector regions with explicit storage effects.
struct CVOutlineVFPass : PassWrapper<CVOutlineVFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVOutlineVFPass)
  StringRef getArgument() const final { return "pto-cv-outline-vf"; }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    // Structural ordinals are independent of textual SSA names. The identity
    // survives outlining, slot expansion and final physical binding.
    module.walk([&](func::FuncOp function) {
      unsigned ordinal = 0;
      function.walk([&](AllocTileOp allocation) {
        if (!allocation->hasAttr("pto.costmodel.buffer_id"))
          allocation->setAttr("pto.costmodel.buffer_id", StringAttr::get(module.getContext(),
              (function.getSymName() + ":buffer:" + Twine(ordinal)).str()));
        ++ordinal;
      });
      ordinal = 0;
      function.walk([&](scf::ForOp loop) {
        if (!loop->hasAttr("pto.costmodel.loop_id"))
          loop->setAttr("pto.costmodel.loop_id", StringAttr::get(module.getContext(),
              (function.getSymName() + ":loop:" + Twine(ordinal)).str()));
        ++ordinal;
      });
    });
    // Cube fusion regions are grouping metadata, not vector functions. Keep
    // their tile operations visible to the outer CV scheduler.
    SmallVector<FusionRegionOp> cubeRegions;
    SmallVector<Operation *> singletons;
    module.walk([&](Operation *op) {
      auto function = op->getParentOfType<func::FuncOp>();
      if (!function || function->hasAttr("pto.cv.prepared_vf")) return;
      auto kind = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
      if (!kind) return;
      if (kind.getKernelKind() == FunctionKernelKind::Cube) {
        if (auto region = dyn_cast<FusionRegionOp>(op)) cubeRegions.push_back(region);
        return;
      }
      auto pipe = dyn_cast<OpPipeInterface>(op);
      if (!op->getNumRegions() && op->hasAttr("candidates") &&
          isa<TileOpInterface>(op) && pipe &&
          pipe.getPipe() == PIPE::PIPE_V &&
          !op->getParentOfType<FusionRegionOp>() &&
          !op->getParentOfType<VecScopeOp>() &&
          !op->getParentOfType<StrictVecScopeOp>()) singletons.push_back(op);
    });
    for (FusionRegionOp region : llvm::reverse(cubeRegions)) {
      auto yield = dyn_cast<pto::YieldOp>(region.getBody().front().getTerminator());
      if (!yield || yield.getNumOperands() != region.getNumResults())
        return signalPassFailure();
      for (auto [result, value] : llvm::zip(region.getResults(), yield.getOperands()))
        result.replaceAllUsesWith(value);
      for (Operation &op : llvm::make_early_inc_range(region.getBody().front().without_terminator()))
        op.moveBefore(region);
      region.erase();
    }
    // FusionPlan deliberately leaves singleton arithmetic ungrouped. It still
    // needs the same parameterized VF boundary when adjacent pipe operations
    // prevent forming a larger fusion group.
    for (Operation *op : singletons) {
      OpBuilder b(op);
      auto region = b.create<FusionRegionOp>(op->getLoc(), TypeRange{});
      if (region.getBody().empty()) region.getBody().push_back(new Block());
      op->moveBefore(&region.getBody().front(), region.getBody().front().end());
      b.setInsertionPointToEnd(&region.getBody().front());
      b.create<pto::YieldOp>(region.getLoc(), ValueRange{});
    }
    SmallVector<Operation *> regions;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto function = op->getParentOfType<func::FuncOp>();
      if (!function || function->hasAttr("pto.cv.prepared_vf")) return WalkResult::advance();
      auto kind = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
      if (kind && kind.getKernelKind() == FunctionKernelKind::Cube) return WalkResult::skip();
      if (isa<FusionRegionOp, VecScopeOp, StrictVecScopeOp>(op)) {
        regions.push_back(op);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    unsigned ordinal = 0;
    for (Operation *region : regions) {
      if (region->getNumRegions() != 1 ||
          !region->getRegion(0).hasOneBlock()) {
        region->emitError("CV VF requires a single-block resultless vector region");
        return signalPassFailure();
      }
      // FusionRegionGen exposes DPS destinations as alias results and may
      // sink their allocations into the region. Recover explicit caller-owned
      // storage before outlining; no vector register crosses the VF boundary.
      std::function<bool(Value)> hoist = [&](Value value) {
        Operation *definition = value.getDefiningOp();
        if (!definition) {
          auto *owner = value.getParentRegion()->getParentOp();
          return owner != region && !region->isProperAncestor(owner);
        }
        if (!region->isProperAncestor(definition)) return true;
        if (!isa<AllocTileOp, arith::ConstantOp, TReshapeOp, SubViewOp, BitcastOp>(definition)) return false;
        for (Value operand : definition->getOperands()) if (!hoist(operand)) return false;
        definition->moveBefore(region);
        return true;
      };
      SmallVector<AllocTileOp> allocations;
      llvm::SmallDenseSet<Value> outputs;
      if (auto fusion = dyn_cast<FusionRegionOp>(region)) {
        if (auto yield = dyn_cast<pto::YieldOp>(fusion.getBody().front().getTerminator()))
          for (Value value : yield.getOperands()) outputs.insert(value);
      }
      region->walk([&](AllocTileOp allocation) { allocations.push_back(allocation); });
      for (AllocTileOp allocation : allocations) {
        if (!hoist(allocation.getResult())) {
          allocation.emitError("VF scratch cannot be bound outside its computation");
          return signalPassFailure();
        }
        if (!outputs.contains(allocation.getResult()))
          allocation->setAttr("pto.cv.vf_scratch", UnitAttr::get(module.getContext()));
        // Static scratch has no runtime allocation side effect. Give CV and
        // PlanMemory an explicit function-owned object for its write versions.
        if (allocation->getNumOperands() == 0) {
          Operation *anchor = region;
          for (Operation *parent = region->getParentOp(); parent && !isa<func::FuncOp>(parent);
               parent = parent->getParentOp())
            if (isa<scf::ForOp>(parent)) anchor = parent;
          allocation->moveBefore(anchor);
        }
      }
      if (auto fusion = dyn_cast<FusionRegionOp>(region)) {
        auto yield = dyn_cast<pto::YieldOp>(fusion.getBody().front().getTerminator());
        if (!yield || yield.getNumOperands() != fusion.getNumResults()) return signalPassFailure();
        for (auto [result, value] : llvm::zip(fusion.getResults(), yield.getOperands())) {
          if (!isa<TileBufType>(value.getType()) || !hoist(value)) {
            fusion.emitError("CV VF does not support escaping scalar or vector-register results");
            return signalPassFailure();
          }
          result.replaceAllUsesWith(value);
        }
        if (fusion.getNumResults()) {
          OpBuilder b(fusion);
          auto replacement = b.create<FusionRegionOp>(fusion.getLoc(), TypeRange{});
          replacement->setAttrs(fusion->getAttrs());
          replacement.getBody().takeBody(fusion.getBody());
          b.setInsertionPoint(yield);
          b.create<pto::YieldOp>(yield.getLoc(), ValueRange{});
          yield.erase();
          fusion.erase();
          region = replacement;
        }
      } else if (region->getNumResults()) {
        region->emitError("CV VF scope results are unsupported");
        return signalPassFailure();
      }
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(region->getRegions(), captures);
      llvm::SetVector<Value> parameters;
      for (Value value : captures) {
        if (value.getDefiningOp<arith::ConstantOp>()) continue;
        if (auto address = value.getDefiningOp<TileBufAddrOp>())
          parameters.insert(address.getSrc());
        else parameters.insert(value);
      }
      SmallVector<Value> arguments;
      SmallVector<Type> types;
      for (Value value : parameters) {
        if (auto tile = dyn_cast<TileBufType>(value.getType())) {
          auto space = dyn_cast<AddressSpaceAttr>(tile.getMemorySpace());
          if (!space || space.getAddressSpace() != AddressSpace::VEC ||
              llvm::is_contained(tile.getShape(), ShapedType::kDynamic) ||
              llvm::is_contained(tile.getValidShape(), ShapedType::kDynamic)) {
            region->emitError("CV VF requires static vector-local buffers");
            return signalPassFailure();
          }
          auto root = resolveVmiTileAccess(value);
          if (auto allocation = value.getDefiningOp<AllocTileOp>()) {
            if (allocation.getAddr()) {
              IntegerAttr address;
              if (!matchPattern(allocation.getAddr(), m_Constant(&address)) || address.getInt() % 32) {
                region->emitError("CV VF local binding must preserve 32-byte alignment");
                return signalPassFailure();
              }
            }
          }
          // Pipe lowering binds declare_tile to an aligned reserved FIFO slot.
          // It is borrowed storage and remains outside local slot expansion.
          bool borrowed = value.getDefiningOp<DeclareTileOp>() &&
              llvm::count_if(value.getUsers(), [](Operation *user) { return isa<TPopOp>(user); }) == 1;
          if (borrowed) root = VmiTileAccess{value, value, false, 0, std::nullopt, value.getType()};
          if (!root || root->unknownRoot || !root->byteOffset || *root->byteOffset % 32 ||
              (!root->root.getDefiningOp<AllocTileOp>() && !borrowed)) {
            region->emitError("CV VF requires a proven 32-byte aligned local or borrowed FIFO buffer");
            return signalPassFailure();
          }
          for (Value previous : arguments) {
            if (!isa<TileBufType>(previous.getType())) continue;
            auto other = resolveVmiTileAccess(previous);
            if (previous.getDefiningOp<DeclareTileOp>())
              other = VmiTileAccess{previous, previous, false, 0, std::nullopt, previous.getType()};
            if (!root || !other || root->unknownRoot || other->unknownRoot ||
                relateVmiTileAccess(*root, *other) != VmiAccessRelation::NoAlias) {
              region->emitError("CV VF distinct buffer parameters may alias");
              return signalPassFailure();
            }
          }
          types.push_back(PtrType::get(module.getContext(), tile.getElementType(), space));
        } else if (value.getType().isIntOrIndexOrFloat()) {
          types.push_back(value.getType());
        } else {
          region->emitError("CV VF capture requires a static tile or scalar");
          return signalPassFailure();
        }
        arguments.push_back(value);
      }
      auto owner = region->getParentOfType<func::FuncOp>();
      auto container = owner->getParentOfType<ModuleOp>();
      OpBuilder b(container.getContext());
      b.setInsertionPointToEnd(container.getBody());
      std::string name = (owner.getSymName() + "__cv_vf_" + Twine(ordinal++)).str();
      if (container.lookupSymbol(name)) {
        region->emitError("CV VF symbol collision"); return signalPassFailure();
      }
      auto helper = b.create<func::FuncOp>(region->getLoc(), name,
                                         b.getFunctionType(types, {}));
      helper.setPrivate();
      helper->setAttr("no_inline", b.getUnitAttr());
      helper->setAttr("pto.cv.prepared_vf", b.getStringAttr("1.0"));
      helper->setAttr("pto.cv.distinct_buffers", b.getUnitAttr());
      helper->setAttr("pto.cv.argument_alignment_bytes", b.getI64IntegerAttr(32));
      helper->setAttr("pto.kernel_kind", FunctionKernelKindAttr::get(
          b.getContext(), FunctionKernelKind::Vector));
      Block *entry = helper.addEntryBlock();
      b.setInsertionPointToStart(entry);
      IRMapping mapping;
      for (auto [value, argument] : llvm::zip(arguments, entry->getArguments())) {
        if (isa<TileBufType>(value.getType())) {
          Value address = b.create<CastPtrOp>(region->getLoc(), b.getI64Type(), argument);
          OperationState state(region->getLoc(), AllocTileOp::getOperationName());
          state.addOperands(address); state.addTypes(value.getType());
          state.addAttribute("operandSegmentSizes", b.getDenseI32ArrayAttr({1, 0, 0}));
          state.addAttribute("pto.cv.borrowed", b.getUnitAttr());
          mapping.map(value, b.create(state)->getResult(0));
        } else mapping.map(value, argument);
      }
      for (Value value : captures)
        if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
          b.clone(*constant, mapping);
        } else if (auto address = value.getDefiningOp<TileBufAddrOp>()) {
          mapping.map(value, b.create<TileBufAddrOp>(region->getLoc(), value.getType(), mapping.lookup(address.getSrc())));
        }
      b.clone(*region, mapping);
      b.create<func::ReturnOp>(region->getLoc());
      b.setInsertionPoint(region);
      SmallVector<Value> operands;
      for (auto [value, type] : llvm::zip(arguments, types)) {
        if (isa<TileBufType>(value.getType()))
          operands.push_back(b.create<TileBufAddrOp>(region->getLoc(), type, value));
        else operands.push_back(value);
      }
      b.create<func::CallOp>(region->getLoc(), name, TypeRange{}, operands);
      region->erase();
    }
  }
};

struct CVValidateVFPass : PassWrapper<CVValidateVFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVValidateVFPass)
  StringRef getArgument() const final { return "pto-cv-validate-vf"; }
  void runOnOperation() final {
    bool invalid = false;
    getOperation().walk([&](func::FuncOp function) {
      if (!function->hasAttr("pto.cv.prepared_vf")) return;
      auto effects = getCVVFEffects(function);
      if (failed(effects)) { invalid = true; return; }
      Builder b(function.getContext());
      SmallVector<Attribute> summary;
      for (unsigned i = 0; i < effects->reads.size(); ++i)
        summary.push_back(b.getStringAttr(effects->reads[i]
            ? (effects->writes[i] ? "readwrite" : "read")
            : (effects->writes[i] ? "write" : "none")));
      function->setAttr("pto.tileop.effects", b.getArrayAttr(summary));
    });
    getOperation().walk([&](func::CallOp call) {
      auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
      if (!function || !function->hasAttr("pto.cv.prepared_vf")) return;
      llvm::SmallDenseSet<Value> roots;
      for (Value value : call.getOperands()) {
        if (!isa<PtrType>(value.getType())) continue;
        Value root = getCVStorageRoot(value);
        if (!root || !roots.insert(root).second) {
          call.emitError("prepared VF distinct buffer contract is not satisfied");
          invalid = true;
        }
      }
    });
    if (invalid) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createCVOutlineVFPass() {
  return std::make_unique<CVOutlineVFPass>();
}
std::unique_ptr<Pass> mlir::pto::createCVValidateVFPass() {
  return std::make_unique<CVValidateVFPass>();
}
