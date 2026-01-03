# Design Document: ArgCompareOp Support in VectorDistribute Pipeline

## Table of Contents
- [Executive Summary](#executive-summary)
- [Current Task: Decomposition](#current-task-decomposition)
- [Background](#background)
- [Two Approaches](#two-approaches)
- [Implementation Plan](#implementation-plan)
- [Key Files Reference](#key-files-reference)
- [Open Questions](#open-questions)

---

## Executive Summary

This document outlines the design for supporting `iree_linalg_ext.arg_compare` operations through IREE's VectorDistribute pipeline, with a focus on AMD GPU backends.

### Two Possible Approaches

1. **Decomposition Approach (Current Focus)**: Decompose `ArgCompareOp` to `linalg.generic` matching `isArgmaxOp()` pattern, reusing existing infrastructure
2. **Direct Vectorization Approach**: Implement custom vectorization and distribution for `ArgCompareOp`

We are starting with the **Decomposition Approach** as it requires less new code and leverages existing battle-tested infrastructure.

---

## Current Task: Decomposition

### Immediate Goal

Add `AggregatedOpInterface` to `ArgCompareOp` and implement `decomposeOperation()` to convert recognized argmax/argmin patterns to `linalg.generic`.

### Why Decomposition First?

| Aspect | Direct Vectorization | Decomposition Approach |
|--------|---------------------|------------------------|
| **New Interfaces** | Need to modify op to support vector types | Just add `AggregatedOpInterface` |
| **Vectorization Code** | Custom vectorizer needed | Reuse existing `linalg::vectorize()` |
| **GPU Distribution** | Custom `DistributeArgCompare` pattern | Reuse existing `DistributeMultiReduction` |
| **Split-K** | Need new split implementation | Reuse existing `splitArgmaxReduction()` |
| **Testing** | All new test coverage | Mostly covered by existing argmax tests |
| **Maintenance** | New code paths to maintain | Reuses battle-tested code |

### Pipeline After Decomposition

```
iree_linalg_ext.arg_compare
    ↓ (DecomposeAggregatedOps - decomposeOperation)
linalg.generic (matching isArgmaxOp pattern)
    ↓ (splitArgmaxReduction - if needed for split-K)
linalg.generic (partial) + linalg.generic (final)
    ↓ (linalg::vectorize)
vector.multi_reduction + index tracking ops
    ↓ (LLVMGPUVectorDistribute - DistributeMultiReduction)
gpu.subgroup_reduce + ballot + readlane
    ↓ (GPU lowering)
DPP + AMD intrinsics
```

---

## Background

### ArgCompareOp Overview

`ArgCompareOp` is defined in [LinalgExtOps.td:645-770](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td#L645-L770) and performs:
- A reduction over a specified dimension of a tensor
- Returns **two outputs**: the selected value AND its corresponding index
- Uses a **user-defined comparator region** that receives two values and returns `i1`

**Key Design Feature**: The comparator region provides flexibility to express:
- **argmax**: `arith.cmpf ogt, %a, %b` (greater than)
- **argmin**: `arith.cmpf olt, %a, %b` (less than)
- **Custom logic**: Any boolean predicate comparing two values

```mlir
// Example: argmax (select larger value)
iree_linalg_ext.arg_compare dimension(1)
  ins(%input : tensor<2x10xf32>)
  outs(%out_val, %out_idx : tensor<2xf32>, tensor<2xi32>) {
^bb0(%a: f32, %b: f32):
  %cmp = arith.cmpf ogt, %a, %b : f32
  iree_linalg_ext.yield %cmp : i1
}

// Example: argmin (select smaller value)
iree_linalg_ext.arg_compare dimension(1)
  ins(%input : tensor<2x10xf32>)
  outs(%out_val, %out_idx : tensor<2xf32>, tensor<2xi32>) {
^bb0(%a: f32, %b: f32):
  %cmp = arith.cmpf olt, %a, %b : f32
  iree_linalg_ext.yield %cmp : i1
}

// Example: custom comparator (select value with larger absolute value)
iree_linalg_ext.arg_compare dimension(1)
  ins(%input : tensor<2x10xf32>)
  outs(%out_val, %out_idx : tensor<2xf32>, tensor<2xi32>) {
^bb0(%a: f32, %b: f32):
  %abs_a = math.absf %a : f32
  %abs_b = math.absf %b : f32
  %cmp = arith.cmpf ogt, %abs_a, %abs_b : f32
  iree_linalg_ext.yield %cmp : i1
}
```

### Existing ArgMax Infrastructure

There's already comprehensive infrastructure for `linalg.generic` based argmax:

1. **ArgMax Detection**: [Utils.cpp:904-959](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L904-L959)
   - `isArgmaxOp()` detects argmax pattern in `linalg.generic`
   - Requires: 1 input, 2 outputs, 1 reduction dim, neg-inf init, `maximumf`/`cmpf`/`select` pattern

2. **Split Reduction**: [SplitReduction.cpp:527-738](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp#L527-L738)
   - `splitArgmaxReduction()` for split-K support
   - Handles the complex index tracking during split

3. **UKernel Path**: [iree_uk_amdgpu_argmax_f32i64.c](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c)
   - Algorithm reference using ballot + DPP:
   ```c
   // 1. Reduce to find maximum across subgroup
   float wgMax = laneMax;
   for (int i = 1; i < warpSize; i *= 2) {
     wgMax = __builtin_fmaxf(__shfl_xor_f(wgMax, i), wgMax);
   }

   // 2. Use ballot to find which lanes have the max
   uint64_t laneHasMaxValmask = __ballot(wgMax == laneMax);

   // 3. Handle index selection (tie-breaking: smallest index wins)
   if (__builtin_popcountll(laneHasMaxValmask) == 1) {
     if (wgMax == laneMax) outputBufferIdx[offset] = laneResult;
   } else {
     int64_t indexVal = wgMax == laneMax ? laneResult : INT64_MAX;
     laneResult = __ockl_wfred_min_i64(indexVal);
     if (laneID == 0) outputBufferIdx[offset] = laneResult;
   }
   ```

### Current VectorDistribute Pipeline for Reductions

```
linalg.generic → vector.multi_reduction → gpu.subgroup_reduce → amdgpu.dpp → rocdl.update.dpp
```

Key pattern: `DistributeMultiReduction` in [GPUNestedLayoutDistributionPatterns.cpp:966-1395](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp#L966-L1395)

---

## Two Approaches

### Approach 1: Decomposition (Current Focus)

Decompose `ArgCompareOp` to `linalg.generic` matching `isArgmaxOp()` pattern.

**Pros:**
- Reuses existing infrastructure
- Minimal new code
- Automatically benefits from future argmax improvements

**Cons:**
- Only works for recognized patterns (argmax/argmin)
- Custom comparators fall back to scalar path

### Approach 2: Direct Vectorization (Future Work)

Implement custom vectorization and distribution for `ArgCompareOp`.

**Pipeline:**
```
iree_linalg_ext.arg_compare
    ↓ (KernelConfig: LLVMGPUVectorDistribute)
    ↓ (GenericVectorizationPass - vectorizeArgCompareOp)
vectorized arg_compare (value + index vectors)
    ↓ (LLVMGPUConfigureTensorLayouts - setArgCompareAnchor)
iree_vector_ext.to_layout with NestedLayoutAttr
    ↓ (VectorLayoutAnalysis - propagate layouts)
all vector values have layouts
    ↓ (LLVMGPUVectorDistribute - DistributeArgCompare)
    ├─► Standard argmax/argmin:
    │   gpu.subgroup_reduce + rocdl.ballot + rocdl.readlane
    └─► Custom comparator:
        Inline DPP stages with cloned comparator
    ↓ (GPU/AMDGPU Lowering)
amdgpu.dpp + rocdl.ballot + rocdl.readlane
    ↓ (ROCDLToLLVM)
llvm.amdgcn.update.dpp + llvm.amdgcn.ballot + llvm.amdgcn.readlane
```

**Standard ArgMax/ArgMin Generated IR:**
```mlir
// Step 1: Value reduction using existing gpu.subgroup_reduce
%max_val = gpu.subgroup_reduce maximumf %local_val cluster(size = 64) : (f32) -> f32

// Step 2: Find which lanes have the maximum value
%has_max = arith.cmpf oeq, %local_val, %max_val : f32

// Step 3: Ballot to get bitmask of lanes with max
%ballot_mask = rocdl.ballot %has_max : (i1) -> i64

// Step 4: Find first lane with max (smallest index for tie-breaking)
%winner_lane_i64 = llvm.intr.cttz(%ballot_mask, %false) : (i64, i1) -> i64
%winner_lane = arith.trunci %winner_lane_i64 : i64 to i32

// Step 5: Read the index from the winner lane
%final_idx = rocdl.readlane %local_idx, %winner_lane : (i32, i32) -> i32
```

---

## Current Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| **Op Definition** | ✅ Complete | [LinalgExtOps.td:645-770](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td#L645-L770) |
| **Verification** | ✅ Complete | [LinalgExtOps.cpp:1154-1242](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp#L1154-L1242) |
| **TilingInterface** | ✅ Complete | [TilingInterfaceImpl.cpp:1333-1698](compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp#L1333-L1698) |
| **PartialReductionOpInterface** | ✅ Complete | [TilingInterfaceImpl.cpp:1486-1698](compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp#L1486-L1698) |
| **Loop Lowering** | ✅ Complete | [ConvertToLoops.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/ConvertToLoops.cpp) |
| **AggregatedOpInterface** | ❌ Missing | Need to add interface + `decomposeOperation()` |
| **isArgmaxOp detection** | ✅ Exists | [Utils.cpp:904-959](compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L904-L959) |
| **splitArgmaxReduction** | ✅ Exists | [SplitReduction.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp) |

---

## Proposed Solution

### High-Level Approach: Decomposition via AggregatedOpInterface

```
iree_linalg_ext.arg_compare
    ↓ [decomposeOperation() - NEW]
Match comparator pattern:
  - arith.cmpf ogt/oge → argmax
  - arith.cmpf olt/ole → argmin
    ↓
linalg.generic (argmax/argmin pattern matching isArgmaxOp())
    ↓ [Existing Infrastructure]
- isArgmaxOp() detection ✅
- splitArgmaxReduction() for split-K ✅
- linalg::vectorize() ✅
- DistributeMultiReduction + ballot/readlane for indices ✅
```

### Key Design Decisions

1. **Follow AttentionOp Pattern**: Use `AggregatedOpInterface` with `decomposeOperation()` method

2. **Pattern Match Comparator**: Analyze the comparator region to detect:
   - **ArgMax**: `arith.cmpf ogt/oge` or `arith.cmpi sgt/ugt`
   - **ArgMin**: `arith.cmpf olt/ole` or `arith.cmpi slt/ult`
   - **Custom**: Unsupported initially (fallback to scalar path)

3. **Generate linalg.generic Matching isArgmaxOp()**: The decomposed generic must have:
   - 1 input, 2 outputs (value + index)
   - 1 reduction dimension
   - `linalg.fill` with `-inf` for max value init (or `+inf` for min)
   - Body: `arith.maximumf` + `arith.cmpf` + `arith.select` + `linalg.index`

4. **Reuse DecomposeAggregatedOps Pass**: No new pass needed - add `arg_compare` to existing pass filter

---

## Detailed Design

### Step 1: Add AggregatedOpInterface to ArgCompareOp

```tablegen
// In LinalgExtOps.td, modify ArgCompareOp definition
def IREELinalgExt_ArgCompareOp : IREELinalgExt_Op<"arg_compare", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
  DeclareOpInterfaceMethods<LinalgFusionInterface, [...]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,  // ADD THIS
  DeclareOpInterfaceMethods<TilingInterface, [...]>,
  DeclareOpInterfaceMethods<PartialReductionOpInterface, [...]>
]>
```

### Step 2: Implement decomposeOperation()

Add to `AggregatedOpInterfaceImpl.cpp`:

```cpp
//===----------------------------------------------------------------------===//
// ArgCompareOp
//===----------------------------------------------------------------------===//

/// Analyze comparator region to determine if it's argmax or argmin
enum class ArgCompareKind { ArgMax, ArgMin, Custom };

static ArgCompareKind analyzeComparator(Region &comparator) {
  Block &block = comparator.front();

  // Simple case: exactly 2 ops (cmp + yield)
  if (block.getOperations().size() != 2)
    return ArgCompareKind::Custom;

  Operation *cmpOp = &block.front();

  // Check for arith.cmpf
  if (auto cmpf = dyn_cast<arith::CmpFOp>(cmpOp)) {
    // Verify operands are block arguments in correct order
    if (cmpf.getLhs() != block.getArgument(0) ||
        cmpf.getRhs() != block.getArgument(1))
      return ArgCompareKind::Custom;

    switch (cmpf.getPredicate()) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGT:
    case arith::CmpFPredicate::UGE:
      return ArgCompareKind::ArgMax;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULT:
    case arith::CmpFPredicate::ULE:
      return ArgCompareKind::ArgMin;
    default:
      return ArgCompareKind::Custom;
    }
  }

  // Check for arith.cmpi (integer comparison)
  if (auto cmpi = dyn_cast<arith::CmpIOp>(cmpOp)) {
    if (cmpi.getLhs() != block.getArgument(0) ||
        cmpi.getRhs() != block.getArgument(1))
      return ArgCompareKind::Custom;

    switch (cmpi.getPredicate()) {
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::uge:
      return ArgCompareKind::ArgMax;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule:
      return ArgCompareKind::ArgMin;
    default:
      return ArgCompareKind::Custom;
    }
  }

  return ArgCompareKind::Custom;
}

FailureOr<SmallVector<Value>> ArgCompareOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();

  // Analyze comparator to determine if argmax or argmin
  ArgCompareKind kind = analyzeComparator(getRegion());
  if (kind == ArgCompareKind::Custom) {
    // Cannot decompose custom comparators - they'll use scalar lowering path
    return failure();
  }

  Value input = getInputValue();
  Value outVal = outputValue();
  Value outIdx = outputIndex();
  int64_t reductionDim = getDimension();

  auto inputType = cast<RankedTensorType>(input.getType());
  auto outValType = cast<RankedTensorType>(outVal.getType());
  auto outIdxType = cast<RankedTensorType>(outIdx.getType());

  Type elemType = inputType.getElementType();
  Type idxElemType = outIdxType.getElementType();
  int64_t rank = inputType.getRank();

  // Create identity value for reduction
  // ArgMax: -inf, ArgMin: +inf
  Value identityVal;
  if (isa<FloatType>(elemType)) {
    APFloat infVal = APFloat::getInf(
        cast<FloatType>(elemType).getFloatSemantics(),
        /*Negative=*/(kind == ArgCompareKind::ArgMax));
    identityVal = b.create<arith::ConstantOp>(
        loc, elemType, b.getFloatAttr(elemType, infVal));
  } else {
    // Integer type
    unsigned bitWidth = elemType.getIntOrFloatBitWidth();
    int64_t initVal = (kind == ArgCompareKind::ArgMax)
                          ? std::numeric_limits<int64_t>::min()
                          : std::numeric_limits<int64_t>::max();
    identityVal = b.create<arith::ConstantOp>(
        loc, elemType, b.getIntegerAttr(elemType, initVal));
  }

  // Initial index value (0)
  Value zeroIdx = b.create<arith::ConstantOp>(
      loc, idxElemType, b.getIntegerAttr(idxElemType, 0));

  // Create filled output tensors
  Value filledVal = b.create<linalg::FillOp>(loc, identityVal, outVal).getResult(0);
  Value filledIdx = b.create<linalg::FillOp>(loc, zeroIdx, outIdx).getResult(0);

  // Build indexing maps
  // Input: identity map (all dims)
  // Outputs: project out reduction dim
  SmallVector<AffineExpr> inputExprs, outputExprs;
  for (int64_t i = 0; i < rank; ++i) {
    inputExprs.push_back(b.getAffineDimExpr(i));
    if (i != reductionDim) {
      outputExprs.push_back(b.getAffineDimExpr(i));
    }
  }

  AffineMap inputMap = AffineMap::get(rank, 0, inputExprs, b.getContext());
  AffineMap outputMap = AffineMap::get(rank, 0, outputExprs, b.getContext());

  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap, outputMap};

  // Iterator types: all parallel except reduction dim
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
  iteratorTypes[reductionDim] = utils::IteratorType::reduction;

  // Create linalg.generic matching isArgmaxOp() pattern
  auto genericOp = b.create<linalg::GenericOp>(
      loc,
      /*resultTypes=*/TypeRange{outValType, outIdxType},
      /*inputs=*/ValueRange{input},
      /*outputs=*/ValueRange{filledVal, filledIdx},
      indexingMaps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // args: [inputElem, accVal, accIdx]
        Value inputElem = args[0];
        Value accVal = args[1];
        Value accIdx = args[2];

        // Get current index
        Value currentIdxIndex = nestedBuilder.create<linalg::IndexOp>(
            nestedLoc, reductionDim);
        Value currentIdx = nestedBuilder.create<arith::IndexCastOp>(
            nestedLoc, idxElemType, currentIdxIndex);

        Value newVal, cmp;
        if (isa<FloatType>(elemType)) {
          // Use maximumf/minimumf for float
          if (kind == ArgCompareKind::ArgMax) {
            newVal = nestedBuilder.create<arith::MaximumFOp>(
                nestedLoc, inputElem, accVal);
            cmp = nestedBuilder.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, inputElem, accVal);
          } else {
            newVal = nestedBuilder.create<arith::MinimumFOp>(
                nestedLoc, inputElem, accVal);
            cmp = nestedBuilder.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OLT, inputElem, accVal);
          }
        } else {
          // Use maxsi/minsi for integer
          if (kind == ArgCompareKind::ArgMax) {
            newVal = nestedBuilder.create<arith::MaxSIOp>(
                nestedLoc, inputElem, accVal);
            cmp = nestedBuilder.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::sgt, inputElem, accVal);
          } else {
            newVal = nestedBuilder.create<arith::MinSIOp>(
                nestedLoc, inputElem, accVal);
            cmp = nestedBuilder.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::slt, inputElem, accVal);
          }
        }

        // Select new index if current element wins
        Value newIdx = nestedBuilder.create<arith::SelectOp>(
            nestedLoc, cmp, currentIdx, accIdx);

        nestedBuilder.create<linalg::YieldOp>(
            nestedLoc, ValueRange{newVal, newIdx});
      });

  return SmallVector<Value>{genericOp.getResult(0), genericOp.getResult(1)};
}
```

### Step 3: Use Existing DecomposeAggregatedOps Pass

The existing `DecomposeAggregatedOps` pass in [DecomposeAggregatedOps.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/DecomposeAggregatedOps.cpp) already walks over all `AggregatedOpInterface` ops:

```cpp
getOperation().walk([&](linalg::AggregatedOpInterface aggregatedOp) {
  if (!filter.contains(aggregatedOp->getName().getStringRef())) {
    return;
  }
  FailureOr<SmallVector<Value>> results = aggregatedOp.decomposeOperation(rewriter);
  if (succeeded(results)) {
    rewriter.replaceOp(aggregatedOp, *results);
  }
});
```

Add `"iree_linalg_ext.arg_compare"` to the filter in the pipeline where decomposition should occur.

### Step 4: Verify Decomposed Generic Matches isArgmaxOp()

The decomposed `linalg.generic` must match the pattern in [Utils.cpp:904-959](compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L904-L959):

```
✓ 2 outputs (value + index)
✓ 1 input
✓ 1 reduction dimension
✓ linalg.fill with -inf (or +inf for argmin)
✓ Body contains: maximumf (or minimumf), cmpf, select with linalg.index
```

**Note**: Current `isArgmaxOp()` only checks for argmax pattern. We may need to add `isArgminOp()` or generalize to `isArgReduceOp()`.

---

## IR Transformation Example

### Before Decomposition

```mlir
%result:2 = iree_linalg_ext.arg_compare dimension(1)
    ins(%input : tensor<4x128xf32>)
    outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
} -> tensor<4xf32>, tensor<4xi32>
```

### After Decomposition

```mlir
// Fill with identity values
%neg_inf = arith.constant dense<0xFF800000> : f32  // -inf
%zero_idx = arith.constant 0 : i32
%filled_val = linalg.fill ins(%neg_inf : f32) outs(%out_val : tensor<4xf32>) -> tensor<4xf32>
%filled_idx = linalg.fill ins(%zero_idx : i32) outs(%out_idx : tensor<4xi32>) -> tensor<4xi32>

// Decomposed linalg.generic matching isArgmaxOp() pattern
%result:2 = linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,  // input
        affine_map<(d0, d1) -> (d0)>,       // out_val
        affine_map<(d0, d1) -> (d0)>        // out_idx
    ],
    iterator_types = ["parallel", "reduction"]
} ins(%input : tensor<4x128xf32>)
  outs(%filled_val, %filled_idx : tensor<4xf32>, tensor<4xi32>) {
^bb0(%in: f32, %acc_val: f32, %acc_idx: i32):
    // Get current index along reduction dimension
    %idx = linalg.index 1 : index
    %idx_i32 = arith.index_cast %idx : index to i32

    // Compute new max
    %new_max = arith.maximumf %in, %acc_val : f32

    // Compare to see if current element is the new max
    %cmp = arith.cmpf ogt, %in, %acc_val : f32

    // Select index: use current index if we found new max
    %new_idx = arith.select %cmp, %idx_i32, %acc_idx : i32

    linalg.yield %new_max, %new_idx : f32, i32
} -> tensor<4xf32>, tensor<4xi32>
```

### After Existing Infrastructure Processes It

The decomposed `linalg.generic`:
1. Is detected by `isArgmaxOp()` ✅
2. Can be split via `splitArgmaxReduction()` ✅
3. Is vectorized via `linalg::vectorize()` ✅
4. Is distributed via existing GPU patterns ✅

---

## Implementation Plan

### Phase 1: Add AggregatedOpInterface

1. **Modify LinalgExtOps.td**
   - Add `DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>` to ArgCompareOp

2. **Implement decomposeOperation() in AggregatedOpInterfaceImpl.cpp**
   - Add `analyzeComparator()` helper to detect argmax/argmin patterns
   - Implement `ArgCompareOp::decomposeOperation()` to generate linalg.generic
   - Return `failure()` for custom comparators (they continue to use scalar path)

### Phase 2: Verify Integration with Existing Infrastructure

3. **Verify isArgmaxOp() compatibility**
   - Ensure decomposed generic matches expected pattern
   - May need to extend `isArgmaxOp()` to also handle argmin patterns

4. **Test with existing passes**
   - Run DecomposeAggregatedOps pass on arg_compare ops
   - Verify vectorization works on decomposed generic
   - Verify GPU distribution works

### Phase 3: Pipeline Integration

5. **Add to relevant pipelines**
   - Add `"iree_linalg_ext.arg_compare"` to DecomposeAggregatedOps filter where needed
   - Ensure decomposition happens before vectorization

### Phase 4: Testing

6. **Unit tests**
   - Test decomposition of argmax pattern
   - Test decomposition of argmin pattern
   - Test that custom comparators fail gracefully (return failure, use scalar path)

7. **Integration tests**
   - End-to-end GPU tests with decomposed arg_compare
   - Performance comparison with existing ukernel path

---

## Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| **ArgCompareOp Definition** | [LinalgExtOps.td](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td#L645-L770) | Add AggregatedOpInterface |
| **Decomposition Impl** | [AggregatedOpInterfaceImpl.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp) | Implement decomposeOperation() |
| **Decompose Pass** | [DecomposeAggregatedOps.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/DecomposeAggregatedOps.cpp) | Existing pass - add arg_compare to filter |
| **ArgMax Detection** | [Utils.cpp:904-959](compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L904-L959) | Verify compatibility, may extend for argmin |
| **Split Reduction** | [SplitReduction.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp) | Existing split-K support |
| **AttentionOp (reference)** | [AggregatedOpInterfaceImpl.cpp:405-513](compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp#L405-L513) | Pattern to follow |

---

## How Attention Op is Supported (Reference)

Understanding how `AttentionOp` flows through the VectorDistribute pipeline helps clarify what's needed for `ArgCompareOp`.

### Attention Pipeline Flow

```
AttentionOp (with decomposition_config attribute set by KernelConfig)
    ↓
addGPUVectorDistributePassPipeline():
    ↓ 1. tileAndDistributeToWorkgroup
    ↓ 2. ConvertAttentionToOnlineAttentionPass
    ↓ 3. GPUApplyTilingLevel (Reduction, PartialReduction, Serial)
    ↓ 4. DecomposeAttentionPass
           - Calls decomposeOperation()
           - QK matmul gets qk_attrs (including attention_qk_matmul marker)
           - PV matmul gets pv_attrs (including attention_pv_matmul marker)
    ↓ 5. LLVMGPUConfigureTensorLayoutsPass
           - Finds ops with attention_qk_matmul/attention_pv_matmul attributes
           - Calls setAttentionMatmulAnchor() for special layout handling
    ↓ 6. GenericVectorizationPass
    ↓ 7. LLVMGPUVectorDistributePass
```

### Key Difference: ArgCompare is Simpler

For **AttentionOp**:
- Decomposes to **matmuls** which need special MMA layout handling
- Uses `decomposition_config` with `qk_attrs`/`pv_attrs` to pass lowering configs
- `LLVMGPUConfigureTensorLayouts` has special `setAttentionMatmulAnchor()` function

For **ArgCompareOp**:
- Decomposes to **linalg.generic** (argmax pattern) which is a **reduction**
- **No special attributes needed** - just standard `lowering_config`
- Uses existing reduction layout handling in `LLVMGPUConfigureTensorLayouts`
- Goes through standard `DistributeMultiReduction` pattern

### What This Means for Implementation

1. **No `decomposition_config` attribute needed** for ArgCompareOp
2. **No special layout anchor function needed** (unlike `setAttentionMatmulAnchor`)
3. The decomposed `linalg.generic`:
   - Gets `lowering_config` from tiling passes
   - Layout is handled by existing `setGPULoweringConfigLayout` or `setDerivedThreadConfigLayout`
   - Vectorization uses standard `linalg::vectorize()`
   - Distribution uses existing `DistributeMultiReduction`

This is why the decomposition approach is simpler - we just need to generate a `linalg.generic` that matches `isArgmaxOp()` and the rest flows through existing infrastructure.

---

## Open Questions

1. **isArgmaxOp() and argmin**
   - Current `isArgmaxOp()` only detects argmax pattern
   - Need to verify if we should:
     a. Extend `isArgmaxOp()` to handle both, OR
     b. Add separate `isArgminOp()`, OR
     c. Generalize to `isArgReduceOp()`

2. **Custom comparator fallback**
   - When `decomposeOperation()` returns failure, the op stays as-is
   - Verify the scalar lowering path still works for these cases

3. **Pipeline ordering**
   - Where exactly should DecomposeAggregatedOps run?
   - Before tiling? After tiling? Need to verify with attention's placement

4. **Index type handling**
   - Current impl uses `arith.index_cast` to convert `index` to output index type
   - Verify this works for various index types (i32, i64)

5. **NaN handling for floats**
   - Current impl uses `arith.maximumf` which propagates NaN
   - Is this the desired behavior? Match the original arg_compare semantics

---

## Conclusion

By following the **AttentionOp decomposition pattern**, we can support `ArgCompareOp` in the VectorDistribute pipeline with minimal new code:

1. **Add `AggregatedOpInterface`** to ArgCompareOp
2. **Implement `decomposeOperation()`** to generate `linalg.generic` matching `isArgmaxOp()` pattern
3. **Reuse existing infrastructure**: `splitArgmaxReduction()`, `linalg::vectorize()`, `DistributeMultiReduction`

**Key Benefits:**
- Reuses battle-tested code paths
- Minimal new code (just the decomposition logic)
- Automatically benefits from future improvements to argmax infrastructure
- Clear fallback for custom comparators (scalar path)

**Pipeline Flow:**
```
iree_linalg_ext.arg_compare
    ↓ (DecomposeAggregatedOps - decomposeOperation)
linalg.generic (matching isArgmaxOp pattern)
    ↓ (splitArgmaxReduction - if needed for split-K)
linalg.generic (partial) + linalg.generic (final)
    ↓ (linalg::vectorize)
vector.multi_reduction + index tracking ops
    ↓ (LLVMGPUVectorDistribute - DistributeMultiReduction)
gpu.subgroup_reduce + ballot + readlane
    ↓ (GPU lowering)
DPP + AMD intrinsics
```
