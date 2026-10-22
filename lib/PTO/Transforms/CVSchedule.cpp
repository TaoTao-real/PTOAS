// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "PTO/Transforms/CVVF.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;
using namespace mlir::pto;

namespace {
struct Access { Value root; bool read = false, write = false; };
static FailureOr<SmallVector<Access>> accesses(Operation *op) {
  SmallVector<Access> result;
  auto add = [&](Value value, bool read, bool write) {
    if (!read && !write) return;
    Value root = getCVStorageRoot(value);
    if (root && root.getDefiningOp<AllocTileOp>()) result.push_back({root, read, write});
  };
  if (auto call = dyn_cast<func::CallOp>(op)) {
    auto effects = getCVVFEffects(call);
    if (failed(effects)) return failure();
    for (auto [i, value] : llvm::enumerate(call.getOperands()))
      add(value, effects->reads[i], effects->writes[i]);
  } else if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    for (auto &effect : effects) {
      if (Value value = effect.getValue())
        add(value, isa<MemoryEffects::Read>(effect.getEffect()),
            isa<MemoryEffects::Write>(effect.getEffect()));
      else if (!isa<TPushToAicOp, TPushToAivOp, TPopFromAicOp,
                    TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp, TPushOp, TPopOp, TFreeOp>(op))
        return failure();
    }
  } else if (!isMemoryEffectFree(op) &&
             !isa<DeclareTileOp, TPushToAicOp, TPushToAivOp, TPopFromAicOp,
                  TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp, TPushOp, TPopOp, TFreeOp>(op)) {
    return failure();
  }
  return result;
}
static std::optional<int64_t> constant(Value value) {
  IntegerAttr attr;
  if (matchPattern(value, m_Constant(&attr))) return attr.getInt();
  return std::nullopt;
}

/// A bounded native transformation for independent, single-axis four-stage
/// loops. The compiler rejects recurrence or nested control rather than
/// assuming that an arbitrary FA loop is iteration-independent.
struct CVApplySchedulePass
    : PassWrapper<CVApplySchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVApplySchedulePass)
  bool inferMinimumSlots = false;
  explicit CVApplySchedulePass(bool infer = false) : inferMinimumSlots(infer) {}
  StringRef getArgument() const final { return "pto-cv-apply-schedule"; }
  LogicalResult apply(scf::ForOp loop) {
    auto pAttr = loop->getAttrOfType<IntegerAttr>("pto.cv_preload_count");
    if (!pAttr || pAttr.getInt() < 0 || !loop.getInitArgs().empty())
      return loop.emitError("UNSUPPORTED: CV requires nonnegative preload and no loop-carried SSA values");
    auto lb = constant(loop.getLowerBound()), ub = constant(loop.getUpperBound()),
         step = constant(loop.getStep());
    int64_t extent = 0;
    if (!lb || !ub || !step || *step <= 0 || *ub < *lb ||
        __builtin_sub_overflow(*ub, *lb, &extent) || extent / *step > 256)
      return loop.emitError("UNSUPPORTED: CV requires a bounded static positive-step loop");
    int64_t count = extent / *step + (extent % *step != 0);
    if (count > 256) return loop.emitError("CV trip count exceeds 256");
    if (count == 0) { loop.erase(); return success(); }
    int64_t distance = std::min(pAttr.getInt(), count);
    SmallVector<Operation *> operations;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (op.getNumRegions()) return op.emitError("UNSUPPORTED: nested CV stage control");
      operations.push_back(&op);
    }
    if (operations.size() * count > 65536)
      return loop.emitError("CV expanded candidate exceeds the operation budget");
    // The two core projections use the same transaction split. Explicit stage
    // provenance is also accepted, but it must form a single prefix/suffix cut.
    unsigned split = operations.size();
    unsigned sends = 0, receives = 0;
    auto function = loop->getParentOfType<func::FuncOp>();
    auto kind = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
    bool cube = kind && kind.getKernelKind() == FunctionKernelKind::Cube;
    bool vector = kind && kind.getKernelKind() == FunctionKernelKind::Vector;
    for (auto [i, op] : llvm::enumerate(operations)) {
      if ((isa<TPushToAivOp>(op) || (cube && isa<TPushOp>(op))) && ++sends == 1) split = i + 1;
      if ((isa<TPopFromAicOp>(op) || (vector && isa<TPopOp>(op))) && ++receives == 2) split = i;
      if (auto stage = op->getAttrOfType<StringAttr>("pto.cv.stage"))
        if (stage.getValue() == "suffix") split = std::min<unsigned>(split, i);
    }
    if (split == 0 || split == operations.size())
      return loop.emitError("UNSUPPORTED: missing prefix/suffix transaction boundary");
    DenseMap<Value, bool> written;
    llvm::SetVector<Value> roots;
    SmallVector<SmallVector<Access>> effects;
    for (Operation *op : operations) {
      auto effect = accesses(op);
      if (failed(effect)) return op->emitError("UNSUPPORTED: unknown CV operation effects");
      for (const Access &access : *effect) {
        roots.insert(access.root);
        if (access.read && !written[access.root]) {
          // A read-only input is legal. A later write would turn this into a
          // recurrence; detect that after collecting the complete stage.
          written.try_emplace(access.root, false);
        }
        if (access.write) written[access.root] = true;
      }
      effects.push_back(std::move(*effect));
    }
    for (Value root : roots) {
      bool hasWriter = false;
      for (auto &effect : effects)
        for (const Access &a : effect) if (a.root == root && a.write) hasWriter = true;
      bool seenWriter = false;
      for (auto &effect : effects) {
        for (const Access &a : effect)
          if (a.root == root && a.read && !seenWriter && hasWriter)
            return loop.emitError("UNSUPPORTED_RECURRENCE: local value is read before its iteration writer");
        for (const Access &a : effect)
          if (a.root == root && a.write) seenWriter = true;
      }
    }
    // Preserve the serial program's write generations. Different writes to
    // the same logical buffer need not share one iteration-long allocation.
    // Explicit read-modify-write accesses retain the previous generation's
    // slot: the native operation cannot read and write different pointers.
    struct Version {
      Value root;
      int64_t first = -1, last = -1;
      unsigned slot = 0;
    };
    SmallVector<Version> versions;
    SmallVector<DenseMap<Value, unsigned>> eventVersions(count * operations.size());
    DenseMap<Value, unsigned> current;
    DenseMap<Value, SmallVector<unsigned>> rootVersions;
    for (int64_t i = 0; i < count; ++i) {
      for (unsigned j = 0; j < operations.size(); ++j) {
        DenseMap<Value, std::pair<bool, bool>> combined;
        for (const Access &a : effects[j]) {
          combined[a.root].first |= a.read;
          combined[a.root].second |= a.write;
        }
        for (Value root : roots) {
          if (!combined.count(root)) continue;
          auto effect = combined[root];
          if (!current.count(root) || (effect.second && !effect.first)) {
            unsigned id = versions.size();
            versions.push_back(Version{root});
            rootVersions[root].push_back(id);
            current[root] = id;
          }
          eventVersions[i * operations.size() + j][root] = current[root];
        }
      }
    }
    // Positions come from the compiler's schedule, never predicted time.
    SmallVector<std::pair<unsigned, unsigned>> sequence;
    for (int64_t t = 0; t < count + distance; ++t) {
      if (t < count) for (unsigned j = 0; j < split; ++j) sequence.emplace_back(t, j);
      if (t >= distance) for (unsigned j = split; j < operations.size(); ++j)
        sequence.emplace_back(t - distance, j);
    }
    for (auto [position, item] : llvm::enumerate(sequence)) {
      auto [iteration, operation] = item;
      auto &event = eventVersions[iteration * operations.size() + operation];
      for (Value root : roots) {
        if (!event.count(root)) continue;
        unsigned id = event[root];
        auto &version = versions[id];
        if (version.first < 0) version.first = position;
        version.last = position;
      }
    }
    OpBuilder b(loop);
    DenseMap<Value, Value> multi;
    for (Value root : roots) {
      auto alloc = root.getDefiningOp<AllocTileOp>();
      auto countAttr = alloc->getAttrOfType<IntegerAttr>("pto.pipeline.multi_buffer_count");
      unsigned slots = countAttr ? countAttr.getInt() : 1;
      if ((countAttr && countAttr.getInt() < 1) || slots > kPtoMultiBufferMaxNum)
        return alloc.emitError("invalid CV multi_buffer_count");
      bool hasWriter = written[root];
      auto ids = rootVersions[root];
      llvm::stable_sort(ids, [&](unsigned a, unsigned b) {
        return versions[a].first < versions[b].first;
      });
      SmallVector<int64_t> ends;
      for (unsigned id : ids) {
        auto &version = versions[id];
        auto available = llvm::find_if(ends, [&](int64_t end) { return end < version.first; });
        version.slot = std::distance(ends.begin(), available);
        if (available == ends.end()) ends.push_back(version.last);
        else *available = version.last;
      }
      unsigned needed = std::max<size_t>(1, ends.size());
      if (needed > kPtoMultiBufferMaxNum)
        return alloc.emitError("UNSUPPORTED: required slots exceed the native multi-buffer limit");
      if (inferMinimumSlots) {
        slots = needed;
        alloc->setAttr("pto.pipeline.multi_buffer_count", b.getI64IntegerAttr(slots));
      }
      if (needed > slots)
        return alloc.emitError("INSUFFICIENT_SLOTS: candidate needs ") << needed << " live-version slots";
      alloc->setAttr("pto.cv.required_slots", b.getI64IntegerAttr(needed));
      if (slots == 1) continue;
      if (!hasWriter || alloc.getAddr())
        return alloc.emitError("UNSUPPORTED: multi-buffer requires an unaddressed iteration-local allocation");
      for (OpOperand &use : root.getUses())
        if (!loop->isProperAncestor(use.getOwner()))
          return alloc.emitError("UNSUPPORTED: multi-buffer has a use outside the candidate loop");
      b.setInsertionPoint(alloc);
      auto type = MultiTileBufType::get(b.getContext(), cast<TileBufType>(root.getType()), slots);
      OperationState state(alloc.getLoc(), AllocMultiTileOp::getOperationName());
      state.addTypes(type);
      state.addAttribute("operandSegmentSizes", b.getDenseI32ArrayAttr({0, 0, 0}));
      for (NamedAttribute attr : alloc->getAttrs())
        if (attr.getName() != "operandSegmentSizes") state.addAttribute(attr.getName(), attr.getValue());
      multi[root] = b.create(state)->getResult(0);
    }
    SmallVector<Attribute> stages, storageVersions, events;
    for (int64_t t = 0; t < count + distance; ++t) {
      if (t < count) stages.push_back(b.getDictionaryAttr({
          b.getNamedAttr("phase", b.getStringAttr("prefix")),
          b.getNamedAttr("iteration", b.getI64IntegerAttr(t))}));
      if (t >= distance) stages.push_back(b.getDictionaryAttr({
          b.getNamedAttr("phase", b.getStringAttr("suffix")),
          b.getNamedAttr("iteration", b.getI64IntegerAttr(t - distance))}));
    }
    for (auto [id, version] : llvm::enumerate(versions)) {
      auto identity = version.root.getDefiningOp()->getAttrOfType<StringAttr>("pto.costmodel.buffer_id");
      storageVersions.push_back(b.getDictionaryAttr({
          b.getNamedAttr("buffer_id", identity ? identity : b.getStringAttr("unidentified")),
          b.getNamedAttr("version", b.getI64IntegerAttr(id)),
          b.getNamedAttr("slot", b.getI64IntegerAttr(version.slot)),
          b.getNamedAttr("first_access", b.getI64IntegerAttr(version.first)),
          b.getNamedAttr("last_access", b.getI64IntegerAttr(version.last))}));
    }
    for (auto [position, item] : llvm::enumerate(sequence)) {
      auto [iteration, operation] = item;
      auto sourceId = operations[operation]->getAttrOfType<StringAttr>("pto.costmodel.operation_id");
      events.push_back(b.getDictionaryAttr({
          b.getNamedAttr("position", b.getI64IntegerAttr(position)),
          b.getNamedAttr("iteration", b.getI64IntegerAttr(iteration)),
          b.getNamedAttr("source_operation", sourceId ? sourceId : b.getStringAttr("unidentified")),
          b.getNamedAttr("source_index", b.getI64IntegerAttr(operation)),
          b.getNamedAttr("phase", b.getStringAttr(operation < split ? "prefix" : "suffix"))}));
    }
    auto module = loop->getParentOfType<ModuleOp>();
    SmallVector<Attribute> schedules;
    if (auto existing = module->getAttrOfType<ArrayAttr>("pto.cv.schedules"))
      schedules.append(existing.begin(), existing.end());
    auto loopId = loop->getAttrOfType<StringAttr>("pto.costmodel.loop_id");
    schedules.push_back(b.getDictionaryAttr({
        b.getNamedAttr("loop_id", loopId ? loopId : b.getStringAttr("unidentified")),
        b.getNamedAttr("trip_count", b.getI64IntegerAttr(count)),
        b.getNamedAttr("requested_preload", pAttr),
        b.getNamedAttr("effective_preload", b.getI64IntegerAttr(distance)),
        b.getNamedAttr("stages", b.getArrayAttr(stages)),
        b.getNamedAttr("events", b.getArrayAttr(events)),
        b.getNamedAttr("storage_versions", b.getArrayAttr(storageVersions))}));
    module->setAttr("pto.cv.schedules", b.getArrayAttr(schedules));
    b.setInsertionPoint(loop);
    SmallVector<IRMapping> mappings(count);
    for (unsigned i = 0; i < count; ++i) {
      mappings[i].map(loop.getInductionVar(), b.create<arith::ConstantIndexOp>(loop.getLoc(), *lb + i * *step));
    }
    for (auto [iteration, operation] : sequence) {
      auto &mapping = mappings[iteration];
      if (llvm::any_of(operations[operation]->getResults(), [&](Value value) {
            return multi.count(getCVStorageRoot(value));
          })) {
        if (auto cast = dyn_cast<CastPtrOp>(operations[operation])) {
          if (!isa<PtrType>(cast.getResult().getType()))
            return cast.emitError("UNSUPPORTED: multi-buffer address observed as a scalar");
        }
        // Recreate view/address producers at each consuming memory event.
        continue;
      }
      auto &event = eventVersions[iteration * operations.size() + operation];
      for (Value root : roots) {
        if (!event.count(root) || !multi.count(root)) continue;
        unsigned id = event[root];
        Value slot = b.create<arith::ConstantIndexOp>(loop.getLoc(), versions[id].slot);
        auto selected = b.create<MultiTileGetOp>(loop.getLoc(), root.getType(), multi[root], slot);
        selected->setAttr("pto.cv.iteration", b.getI64IntegerAttr(iteration));
        selected->setAttr("pto.cv.version", b.getI64IntegerAttr(id));
        mapping.map(root, selected.getResult());
      }
      std::function<Value(Value)> bind = [&](Value value) -> Value {
        Value root = getCVStorageRoot(value);
        if (!multi.count(root) || value == root) return mapping.lookupOrDefault(value);
        Operation *definition = value.getDefiningOp();
        if (!definition || !isa<TileBufAddrOp, CastPtrOp, AddPtrOp, TReshapeOp, SubViewOp, BitcastOp>(definition))
          return mapping.lookupOrDefault(value);
        for (Value operand : definition->getOperands()) mapping.map(operand, bind(operand));
        b.clone(*definition, mapping);
        return mapping.lookupOrDefault(value);
      };
      for (Value operand : operations[operation]->getOperands()) mapping.map(operand, bind(operand));
      Operation *clone = b.clone(*operations[operation], mapping);
      clone->setAttr("pto.cv.iteration", b.getI64IntegerAttr(iteration));
      clone->setAttr("pto.cv.phase", b.getStringAttr(operation < split ? "prefix" : "suffix"));
    }
    loop.erase();
    for (auto [root, value] : multi) if (root.use_empty()) root.getDefiningOp()->erase();
    return success();
  }
  void runOnOperation() final {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr("pto.cv_preload_count")) loops.push_back(loop);
    });
    SmallVector<scf::ForOp> cubeLoops, vectorLoops;
    for (scf::ForOp loop : loops) {
      auto function = loop->getParentOfType<func::FuncOp>();
      auto kind = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
      if (!kind) continue;
      if (kind.getKernelKind() == FunctionKernelKind::Cube) cubeLoops.push_back(loop);
      if (kind.getKernelKind() == FunctionKernelKind::Vector) vectorLoops.push_back(loop);
    }
    if (!cubeLoops.empty() && !vectorLoops.empty()) {
      if (cubeLoops.size() != 1 || vectorLoops.size() != 1) {
        getOperation().emitError("UNSUPPORTED: CV requires one paired static iteration axis");
        return signalPassFailure();
      }
      auto cube = cubeLoops.front(), vector = vectorLoops.front();
      if (constant(cube.getLowerBound()) != constant(vector.getLowerBound()) ||
          constant(cube.getUpperBound()) != constant(vector.getUpperBound()) ||
          constant(cube.getStep()) != constant(vector.getStep()) ||
          cube->getAttr("pto.cv_preload_count") != vector->getAttr("pto.cv_preload_count")) {
        getOperation().emitError("CV paired loops require the same domain and preload distance");
        return signalPassFailure();
      }
    }
    for (scf::ForOp loop : loops) if (failed(apply(loop))) return signalPassFailure();
    SmallVector<AllocTileOp> unused;
    getOperation().walk([&](AllocTileOp allocation) {
      if (allocation->use_empty()) unused.push_back(allocation);
    });
    for (AllocTileOp allocation : unused) allocation.erase();
    bool invalid = false;
    getOperation().walk([&](AllocTileOp allocation) {
      auto count = allocation->getAttrOfType<IntegerAttr>("pto.pipeline.multi_buffer_count");
      if (count && count.getInt() > 1) {
        allocation.emitError("UNSUPPORTED: slot configuration was not materialized by a CV loop");
        invalid = true;
      }
    });
    if (invalid) signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass> mlir::pto::createCVApplySchedulePass(bool inferMinimumSlots) {
  return std::make_unique<CVApplySchedulePass>(inferMinimumSlots);
}
