#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

#include "PTO/IR/PTO.h" 

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::pto;

namespace {
struct PTOVFloopGatherPass : public PassWrapper<PTOVFloopGatherPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOVFloopGatherPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, 
                    memref::MemRefDialect, 
                    arith::ArithDialect>();
  }

ForOp getInnermostSCFFor(ForOp forOp) {
  Block &body = *forOp.getBody();
  ForOp innerMostFor = forOp;
  for (Operation &op : body) {
    if (ForOp innerFor = dyn_cast<ForOp>(&op)) {
      innerMostFor = getInnermostSCFFor(innerFor);
      break;
    }
  }
  return innerMostFor;
}

  bool isSCFForConditionEqual(ForOp a, ForOp b) {
  if (!a || !b) return false;
  return a.getLowerBound() == b.getLowerBound() &&
         a.getUpperBound() == b.getUpperBound() &&
         a.getStep() == b.getStep();
}

bool isNestedSCFForStructureEqual(ForOp a, ForOp b) {
  // 第一步：当前层的循环条件必须一致
  if (!isSCFForConditionEqual(a, b)) return false;

  Block &aBody = *a.getBody();
  Block &bBody = *b.getBody();

  ForOp aInnerFor = nullptr, bInnerFor = nullptr;
  // 第二步：找当前层的内层for
  for (Operation &op : aBody) {
    if (ForOp inner = dyn_cast<ForOp>(&op)) { aInnerFor = inner; break; }
  }
  for (Operation &op : bBody) {
    if (ForOp inner = dyn_cast<ForOp>(&op)) { bInnerFor = inner; break; }
  }

  // 情况1：两个都没有内层for → 结构一致
  if (!aInnerFor && !bInnerFor) return true;
  // 情况2：一个有内层for，一个没有 → 结构不一致
  if (!aInnerFor || !bInnerFor) return false;
  // 情况3：都有内层for → 递归判断内层结构
  return isNestedSCFForStructureEqual(aInnerFor, bInnerFor);
}

bool isOptionalTileOp(ForOp forA) {
  ForOp innerA = getInnermostSCFFor(forA);
  Block &innerABody = *innerA.getBody();
  auto &innerAOps = innerABody.getOperations();
  auto startIt = innerAOps.begin();
  auto endIt = std::prev(innerAOps.end());
  for (auto it = startIt; it != endIt; ++it) {
    Operation *originalOp = &*it;
    if (isa<pto::TAddOp>(originalOp)) {
      return true;
    }
  }
  return false;
}

bool mergeInnermostSCFFor(ForOp forA, ForOp forB) {
  // 前置条件1：两个嵌套for的结构必须完全一致（层数+每层条件）
  if (!isNestedSCFForStructureEqual(forA, forB)) {
    return false;
  }

  // 前置条件2：获取两个嵌套for的【最内层for循环】
  ForOp innerA = getInnermostSCFFor(forA);
  ForOp innerB = getInnermostSCFFor(forB);
  if (!innerA || !innerB) return false;

  // 前置条件3：最内层的循环条件必须一致（兜底判断）
  if (!isSCFForConditionEqual(innerA, innerB)) {
    return false;
  }

  // ========== 核心操作：合并最内层的循环体 ==========
  Block &innerABody = *innerA.getBody();
  Block &innerBBody = *innerB.getBody();
  auto &innerAOps = innerABody.getOperations();
  auto &innerBOps = innerBBody.getOperations();
  OpBuilder builder(forA->getContext());
  
  auto startIt = innerBOps.begin();
  auto endIt = std::prev(innerBOps.end());
  llvm::DenseMap<mlir::Value, mlir::Value> value2valueMap;

  for (auto it = startIt; it != endIt; ++it) {
    Operation *originalOp = &*it;
    
    if (isa<memref::SubViewOp>(originalOp)) {
        Location loc = innerABody.front().getLoc();
        builder.setInsertionPointToStart(&innerABody); 
        memref::SubViewOp originalMemRef = cast<memref::SubViewOp>(originalOp);
        memref::SubViewOp baseMemRef = cast<memref::SubViewOp>(innerABody.front());
        Value subView = builder.create<memref::SubViewOp>(
                loc, cast<MemRefType>(originalMemRef.getType()), originalMemRef.getSource(), 
                baseMemRef.getMixedOffsets(), originalMemRef.getMixedSizes(), originalMemRef.getMixedStrides());
        value2valueMap[originalMemRef->getResult(0)] = subView;
    }
    if (isa<pto::TAddOp>(originalOp)) {
        Operation *Endop = &*std::prev(innerABody.end());
        Location loc = Endop->getLoc();
        builder.setInsertionPoint(Endop); // 插入位置：目标Block的末尾
        pto::TAddOp AddFDps = cast<pto::TAddOp>(originalOp);
        builder.create<pto::TAddOp>(
          loc,
          TypeRange{}, 
          value2valueMap[AddFDps.getSrc0()], value2valueMap[AddFDps.getSrc1()], // ins
          value2valueMap[AddFDps.getDst()]                    // outs
      );
    }
  }
  // ========== 安全删除：第二个整套的嵌套for循环 ==========
  forB.erase();

  return true;
}

void runOnOperation() override {
   func::FuncOp funcOp = getOperation();
  //VFInplaceReuseAnalysis vfInplaceReuseAnalysis(moduleOp);
    for (Block &block : funcOp.getBody()) {
      // 收集当前Block内所有的顶层scf.for（嵌套for的根节点）
      llvm::SmallVector<ForOp, 4> topLevelForOps;
      for (Operation &op : block) {
        if (ForOp forOp = dyn_cast<ForOp>(&op)) {
          topLevelForOps.push_back(forOp);
        }
      }

      // 两两配对，合并符合条件的嵌套for
      for (size_t i = 0; i < topLevelForOps.size(); ++i) {
        ForOp baseFor = topLevelForOps[i];
        if (!baseFor) continue;
        if (!isOptionalTileOp(baseFor)) {
          continue;
        }

        for (size_t j = i + 1; j < topLevelForOps.size(); ++j) {
          ForOp currFor = topLevelForOps[j];
          if (!currFor) continue;
          if (!isOptionalTileOp(currFor)) {
            continue;
          }
          if (currFor->getPrevNode() == baseFor || baseFor->getPrevNode() == currFor) {

            // 调用核心合并函数
            if (mergeInnermostSCFFor(baseFor, currFor)) {
                topLevelForOps[j] = nullptr; // 标记已删除
            }
          }
        }
      }
    }

    llvm::errs() << "\n// === [PTOVFloopGather] Result Dump Start ===\n";
    
    getOperation()->dump();
    
    llvm::errs() << "\n// === [PTOVFloopGather] Result Dump End ===\n\n";
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTOVFloopGatherPass() {
  return std::make_unique<PTOVFloopGatherPass>();
}
} // namespace pto
} // namespace mlir