# Design Document: ArgCompareOp Support in VectorDistribute Pipeline

## Table of Contents
- [Executive Summary](#executive-summary)
- [Current Implementation Status](#current-implementation-status)
- [Background](#background)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Detailed Design](#detailed-design)
- [Handling User-Defined Comparator Regions](#handling-user-defined-comparator-regions)
- [KernelConfig.cpp Configuration](#kernelconfigcpp-configuration)
- [Implementation Plan](#implementation-plan)
- [Key Files Reference](#key-files-reference)
- [Open Questions](#open-questions)

---

## Executive Summary

This document outlines the design for supporting `iree_linalg_ext.arg_compare` operations through IREE's VectorDistribute pipeline, with a focus on AMD GPU backends. The key challenges are:

1. **Dual Output**: `arg_compare` produces two outputs (value and index)
2. **User-Defined Comparator Region**: The operation uses a region-based comparator that can express arbitrary comparison logic beyond simple argmax/argmin

We propose leveraging DPP instructions for value reduction, `rocdl.ballot` operations for efficient index computation, and a strategy for handling both standard and custom comparator regions.

---

## Current Implementation Status

### What Exists Today

| Component | Status | Location |
|-----------|--------|----------|
| **Op Definition** | ✅ Complete | [LinalgExtOps.td:645-770](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td#L645-L770) |
| **Verification** | ✅ Complete | [LinalgExtOps.cpp:1154-1242](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp#L1154-L1242) |
| **TilingInterface** | ✅ Complete | [TilingInterfaceImpl.cpp:1333-1698](compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp#L1333-L1698) |
| **PartialReductionOpInterface** | ✅ Complete | [TilingInterfaceImpl.cpp:1486-1698](compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp#L1486-L1698) |
| **PartitionableLoopsInterface** | ✅ Complete | [PartitionableLoopsInterface.cpp:272-273](compiler/src/iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.cpp#L272-L273) |
| **Loop Lowering** | ✅ Complete | [ConvertToLoops.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/ConvertToLoops.cpp) |
| **Vectorization in GenericVectorizationPass** | ❌ Missing | [GenericVectorization.cpp](compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp) - ArgCompareOp not handled |
| **Layout Configuration (ToLayoutOp insertion)** | ❌ Missing | [LLVMGPUConfigureTensorLayouts.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp) - No anchor for ArgCompareOp |
| **Layout Analysis (propagation)** | ❌ Missing | [VectorLayoutAnalysis.cpp](compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp) - No ArgCompareOp handling |
| **VectorDistribute Pattern** | ❌ Missing | [GPUNestedLayoutDistributionPatterns.cpp](compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp) |
| **KernelConfig for VectorDistribute** | ❌ Missing | [KernelConfig.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp) |

### Existing GPU Support: UKernel Path

There is an existing **ukernel-based argmax** implementation for ROCM:

1. **UKernel Selection**: [LLVMGPUSelectUKernels.cpp:32-41](compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.cpp#L32-L41)
   - Detects argmax pattern via `isArgmaxOp()` on `linalg.generic`
   - Returns ukernel name: `iree_uk_amdgpu_argmax_{elemType}{indexType}`

2. **UKernel Config**: [KernelConfig.cpp:1991-2055](compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp#L1991-L2055)
   - `setArgmaxUkernelConfig()` configures the ukernel pipeline
   - Uses single subgroup per workgroup
   - Each thread reduces `reductionDim/WarpSize` elements

3. **UKernel Implementation**: [iree_uk_amdgpu_argmax_f32i64.c](compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c)
   - Butterfly warp reduction with `__shfl_xor_f`
   - `__ballot` for finding lanes with max value
   - `__ockl_wfred_min_i64` for tie-breaking (smallest index)

4. **ArgMax Detection**: [Utils.cpp:904-959](compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L904-L959)
   - `isArgmaxOp()` detects argmax pattern in `linalg.generic`
   - Requires: 1 input, 2 outputs, 1 reduction dim, neg-inf init, maximumf/cmpf/select pattern

### Key Insight: linalg.generic vs iree_linalg_ext.arg_compare

The ukernel path works on **`linalg.generic`** ops that match the argmax pattern, **not** on `iree_linalg_ext.arg_compare` directly. This means:

- `iree_linalg_ext.arg_compare` is a higher-level abstraction
- It must be either:
  - Lowered to `linalg.generic` to use the ukernel path, OR
  - Vectorized directly through the VectorDistribute pipeline (this design)

---

## Background

### ArgCompareOp Overview

`ArgCompareOp` is defined in [LinalgExtOps.td:645-770](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td#L645-L770) and performs:
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

### How Vectorization Works in IREE

Vectorization in IREE happens primarily through **GenericVectorizationPass** ([GenericVectorization.cpp](compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp)):

```
GenericVectorizationPass
    ↓
Collects candidates: LinalgOp, PadOp, PackOp, UnPackOp, GatherOp
    ↓
For each candidate:
  - LinalgOp → linalg::vectorize() (upstream MLIR)
  - GatherOp → vectorizeLinalgExtGatherToTransferGather()
  - Others → linalg::vectorize() with masking
    ↓
Output: vector.transfer_read/write, vector.multi_reduction, etc.
```

**Key Observations:**
- `linalg::vectorize()` is the upstream MLIR vectorization function
- It handles `linalg::LinalgOp` (generic, matmul, conv, etc.)
- Custom LinalgExt ops like `GatherOp` have dedicated vectorization functions
- **`ArgCompareOp` is NOT in the candidate list** - it's skipped entirely

### How Standard Reductions Work in VectorDistribute

Based on analysis of the VectorDistribute pipeline (see [reduction_vector_distribute_summary.md](reduction_vd/reduction_vector_distribute_summary.md)):

**Full Pipeline Flow:**
```
linalg.generic
    ↓ (GenericVectorizationPass - linalg::vectorize)
vector.multi_reduction
    ↓ (LLVMGPUVectorDistributePass - DistributeMultiReduction)
gpu.subgroup_reduce
    ↓ (gpu-to-amdgpu or expand-subgroup-reduce)
amdgpu.dpp
    ↓ (amdgpu-to-rocdl)
rocdl.update.dpp
```

**Key Pattern: `DistributeMultiReduction`** in [GPUNestedLayoutDistributionPatterns.cpp:966-1395](compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp#L966-L1395)

Performs 4 stages:
1. **Local Thread Reduction**: Each thread reduces its local data (batch × outer × element tiles)
2. **Thread-Level Cross-Reduction**: If `thread_tile > 1` on reduction dims, uses `gpu.subgroup_reduce`
3. **Accumulator Reduction**: Combines with accumulator argument
4. **Subgroup-Level Reduction**: If `subgroup_tile > 1`, uses shared memory + barrier

**Example Transformation** (from check_log.mlir):
```mlir
// Before: vector<64xf32> with nested_layout thread_tile=[64]
%10 = vector.multi_reduction <add>, %9, %cst_0 [0] : vector<64xf32> to f32

// After: Each of 64 threads gets 1 element, then cross-thread reduce
%9 = vector.transfer_read ... : vector<1xf32>
%12 = vector.multi_reduction <add>, %11, %cst [0, 1, 2] : vector<1x1x1xf32> to f32
%13 = gpu.subgroup_reduce add %12 cluster(size = 64) : (f32) -> f32
```

### Existing ArgMax UKernel Algorithm Reference

The ROCM argmax ukernel in [iree_uk_amdgpu_argmax_f32i64.c](compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c) demonstrates the key algorithm:

```c
// 1. Reduce to find maximum across subgroup
float wgMax = laneMax;
for (int i = 1; i < warpSize; i *= 2) {
  wgMax = __builtin_fmaxf(__shfl_xor_f(wgMax, i), wgMax);
}

// 2. Use ballot to find which lanes have the max
uint64_t laneHasMaxValmask = __ballot(wgMax == laneMax);

// 3. Handle index selection
if (__builtin_popcountll(laneHasMaxValmask) == 1) {
  // Single max holder - direct write
  if (wgMax == laneMax) {
    outputBufferIdx[offset] = laneResult;
  }
} else {
  // Multiple max holders - find smallest index (argmax semantics)
  int64_t indexVal = wgMax == laneMax ? laneResult : INT64_MAX;
  laneResult = __ockl_wfred_min_i64(indexVal);
  if (laneID == 0) {
    outputBufferIdx[offset] = laneResult;
  }
}
```

---

## Problem Statement

The current VectorDistribute pipeline does **not** support `ArgCompareOp` because:

1. **Dual Output**: Standard `vector.multi_reduction` produces a single output, but `arg_compare` produces two (value + index)

2. **Index Tracking**: During DPP-based reductions, indices must be tracked alongside values as data moves between lanes

3. **User-Defined Comparator Region**: The comparator region must be inlined/cloned at each reduction stage, which complicates vectorization and DPP lowering

4. **Tie-Breaking**: When multiple lanes have the same selected value (according to the comparator), the smallest index must be selected

5. **No Vectorization in GenericVectorizationPass**: [GenericVectorization.cpp](compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp) is the main vectorization pass in the pipeline. It handles:
   - `linalg::LinalgOp` via upstream `linalg::vectorize()`
   - `tensor::PadOp`, `linalg::PackOp`, `linalg::UnPackOp` (with masking)
   - `IREE::LinalgExt::GatherOp` via custom vectorization
   - **But NOT `ArgCompareOp`** - it's not in the candidate list

6. **No KernelConfig Entry**: `ArgCompareOp` is not routed to the VectorDistribute pipeline in [KernelConfig.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp)

### Challenge: Comparator Region Semantics

The comparator region `^bb0(%a, %b)` returns `true` if `%a` should be preferred over `%b`. This has important implications:

- **Transitivity**: We assume the comparator is transitive (if a > b and b > c, then a > c)
- **Tie Handling**: When comparator returns `false` for both `cmp(a,b)` and `cmp(b,a)`, the values are considered "equal" and tie-breaking by index applies
- **Vectorization Complexity**: The region must be cloned and applied at every comparison point during reduction

---

## Proposed Solution

### High-Level Approach

We propose a **composite reduction** approach that maintains value-index pairs throughout the reduction, following the same pattern as `DistributeMultiReduction`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Vectorization                                                  │
│   ArgCompareOp → vector operations with paired (value, index) vectors   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Distribution (NestedLayoutAttr)                                │
│   Apply same layout to both value and index vectors                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Local Reduction                                                │
│   Each thread reduces its local (value, index) pairs                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Subgroup ArgReduce                                             │
│   Custom DPP-based reduction that tracks both value and index           │
│   Uses ballot for tie-breaking when needed                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **New Operation: `amdgpu.subgroup_arg_compare` with Region** (Option A - Recommended)
   - Introduce a new **AMDGPU dialect** operation that performs arg-reduction with a comparator region
   - Returns both value and index
   - Preserves the user-defined comparator semantics through the lowering
   - Lowers directly to DPP + ballot operations with inlined comparator
   - **Rationale for AMDGPU dialect** (not GPU dialect):
     - Uses AMD-specific DPP instructions (`amdgpu.dpp`) and ballot (`rocdl.ballot`)
     - Follows AMDGPU dialect design principles: wraps AMD-specific functionality
     - Abstracts chipset differences (gfx9 vs gfx10+ have different DPP patterns)
     - The GPU dialect is target-agnostic; this operation is inherently AMD-specific

2. **Decompose to Value Reduce + Ballot** (Option B - Limited)
   - Only works for standard comparators (argmax/argmin)
   - Decompose arg_compare into two phases:
     - Standard value reduction using existing `gpu.subgroup_reduce`
     - Index computation using `rocdl.ballot` + popcount + conditional logic
   - **Limitation**: Cannot handle custom comparator regions

3. **Hybrid Approach** (Option C - Practical)
   - Detect if comparator is a "standard" pattern (simple ogt/olt comparison)
   - For standard patterns: use optimized Option B path
   - For custom patterns: use general Option A path with region inlining

### Why AMDGPU Dialect Instead of GPU Dialect?

| Consideration | GPU Dialect | AMDGPU Dialect |
|---------------|-------------|----------------|
| **Target** | Vendor-agnostic | AMD-specific |
| **Lowering** | Would need intermediate step | Direct to DPP/ballot |
| **DPP operations** | Not available | Native (`amdgpu.dpp`) |
| **Ballot** | Would need `gpu.ballot` (doesn't exist) | Uses `rocdl.ballot` |
| **Chipset handling** | N/A | Can handle gfx9/gfx10+ differences |
| **Design precedent** | `gpu.subgroup_reduce` | `amdgpu.mfma`, `amdgpu.dpp` |

The AMDGPU dialect is the right home because:
1. **The lowering uses AMD-specific primitives** (DPP, ballot, readlane)
2. **Follows existing patterns** - similar to how `amdgpu.mfma` wraps AMD matrix ops
3. **Chipset-aware** - can adapt to hardware differences
4. **Avoids polluting GPU dialect** with vendor-specific operations

---

## Detailed Design

### Option A: New `amdgpu.subgroup_arg_compare` Operation with Comparator Region

#### Operation Definition

```tablegen
def AMDGPU_SubgroupArgCompareOp : AMDGPU_Op<"subgroup_arg_compare", [
    SingleBlockImplicitTerminator<"amdgpu::YieldOp">,
    AttrSizedOperandSegments
]> {
  let summary = "Performs subgroup arg-reduction with user-defined comparator";
  let description = [{
    Reduces values across lanes in a subgroup using a user-defined comparator,
    returning both the selected value and its corresponding index.

    This is an AMD GPU-specific operation that leverages DPP (Data Parallel
    Primitives) instructions for efficient cross-lane communication and
    `rocdl.ballot` for tie-breaking when multiple lanes have equal values.

    The comparator region receives two candidate values (%a, %b) and returns
    an i1 indicating whether %a should be preferred over %b.

    When multiple lanes have "equal" values (comparator returns false for both
    directions), the smallest index is selected (tie-breaking).

    Example (argmax):
    ```mlir
    %val, %idx = amdgpu.subgroup_arg_compare %value, %index
        cluster(size = 64) : (f32, i32) -> (f32, i32) {
      ^bb0(%a: f32, %b: f32):
        %cmp = arith.cmpf ogt, %a, %b : f32
        amdgpu.yield %cmp : i1
    }
    ```

    The lowering uses AMD-specific DPP operations:
    - `amdgpu.dpp` with quad_perm for pairs/quads reduction
    - `amdgpu.dpp` with row_half_mirror/row_mirror for 8/16-lane reduction
    - `rocdl.ballot` + `llvm.cttz` for final winner selection with tie-breaking

    Chipset-aware lowering handles differences between gfx9 (row_bcast) and
    gfx10+ (permlanex16) for cross-row reductions.
  }];

  let arguments = (ins
    AnyTypeOf<[AnyFloat, AnyInteger]>:$value,
    AnyInteger:$index,
    OptionalAttr<I32Attr>:$cluster_size,
    DefaultValuedAttr<I32Attr, "1">:$cluster_stride
  );

  let regions = (region SizedRegion<1>:$comparator);

  let results = (outs
    AnyTypeOf<[AnyFloat, AnyInteger]>:$result_value,
    AnyInteger:$result_index
  );

  let extraClassDeclaration = [{
    /// Returns true if the comparator is a simple argmax pattern (ogt comparison)
    bool isArgMax();
    /// Returns true if the comparator is a simple argmin pattern (olt comparison)
    bool isArgMin();
    /// Returns true if the comparator is a custom (non-standard) pattern
    bool isCustomComparator();
  }];
}
```

#### Lowering `amdgpu.subgroup_arg_compare` to DPP + Ballot

The lowering generates a tree-reduction pattern with the comparator region **inlined at each stage**, following the same DPP patterns used for `gpu.subgroup_reduce` lowering.

**Reference**: [SubgroupReduceLowering.cpp](third_party/llvm-project/mlir/lib/Dialect/GPU/Transforms/SubgroupReduceLowering.cpp) - `createSubgroupDPPReduction`

**DPP Reduction Stages** (for 64-lane subgroup):
| Stage | DPP Mode | Effect |
|-------|----------|--------|
| 1 | `quad_perm([1,0,3,2])` | Pairs: lanes 0↔1, 2↔3, etc. |
| 2 | `quad_perm([2,3,0,1])` | Quads: lanes 0↔2, 1↔3, etc. |
| 3 | `row_half_mirror` | 8-lane: lanes 0↔4, 1↔5, etc. |
| 4 | `row_mirror` | 16-lane: lanes 0↔8, 1↔9, etc. |
| 5 | `row_bcast_15` (gfx9) or `permlanex16` (gfx10+) | Cross-row 32-lane |
| 6 | `row_bcast_31` (gfx9) or `readlane` (gfx10+) | Full 64-lane |

**Generated IR (one DPP stage):**

```mlir
// Stage 1: Reduce pairs (lanes N <-> N+1)
%dpp_val_1 = amdgpu.dpp %value %value quad_perm([1,0,3,2])
    {row_mask = 0xf, bank_mask = 0xf, bound_ctrl = true} : f32
%dpp_idx_1 = amdgpu.dpp %idx %idx quad_perm([1,0,3,2])
    {row_mask = 0xf, bank_mask = 0xf, bound_ctrl = true} : i32

// INLINE COMPARATOR REGION: compare(%value, %dpp_val_1)
%cmp_1 = <inlined comparator result>

// Value and index selection
%val_1 = arith.select %cmp_1, %value, %dpp_val_1 : f32
%idx_1 = arith.select %cmp_1, %idx, %dpp_idx_1 : i32

// Handle tie-breaking (for custom comparators)
%cmp_1_rev = <inlined reverse comparator>
%is_tie = arith.andi (arith.xori %cmp_1, %true), (arith.xori %cmp_1_rev, %true) : i1
%smaller_idx = arith.minsi %idx, %dpp_idx_1 : i32
%idx_1 = arith.select %is_tie, %smaller_idx, %winner_idx : i32
```

**Final Ballot-Based Winner Selection:**
```mlir
// After all DPP stages, use ballot to find the winning lane
%final_val = rocdl.readlane %val_N, %c0 : f32  // Read lane 0's value as reference
%is_winner = <inlined comparator equality check>
%ballot_mask = rocdl.ballot %is_winner : i64
%winner_lane = llvm.intr.cttz(%ballot_mask) : (i64) -> i64
%winner_lane_i32 = arith.trunci %winner_lane : i64 to i32
%final_idx = rocdl.readlane %idx_N, %winner_lane_i32 : i32
```

---

## Handling User-Defined Comparator Regions

### Comparator Region Analysis

Before vectorization, analyze the comparator region to classify it:

```cpp
enum class ComparatorKind {
  ArgMax,       // Single arith.cmpf ogt or arith.cmpi sgt/ugt
  ArgMin,       // Single arith.cmpf olt or arith.cmpi slt/ult
  Custom        // Any other pattern
};

ComparatorKind analyzeComparator(Region &comparator) {
  Block &block = comparator.front();
  if (block.getOperations().size() != 2)  // cmp + yield
    return ComparatorKind::Custom;

  Operation *cmpOp = &block.front();
  if (auto cmpf = dyn_cast<arith::CmpFOp>(cmpOp)) {
    if (cmpf.getPredicate() == arith::CmpFPredicate::OGT &&
        cmpf.getLhs() == block.getArgument(0) &&
        cmpf.getRhs() == block.getArgument(1))
      return ComparatorKind::ArgMax;
    if (cmpf.getPredicate() == arith::CmpFPredicate::OLT &&
        cmpf.getLhs() == block.getArgument(0) &&
        cmpf.getRhs() == block.getArgument(1))
      return ComparatorKind::ArgMin;
  }
  return ComparatorKind::Custom;
}
```

### Strategy by Comparator Kind

| Comparator Kind | Vectorization Strategy | DPP Lowering |
|-----------------|------------------------|--------------|
| **ArgMax** | Use `vector.multi_reduction<maximumf>` + index tracking | Use optimized `arith.maximumf` path |
| **ArgMin** | Use `vector.multi_reduction<minimumf>` + index tracking | Use optimized `arith.minimumf` path |
| **Custom** | Inline comparator at each reduction point | Clone region at each DPP stage |

### Region Cloning During Lowering

```cpp
/// Clone the comparator region and return the comparison result.
Value inlineComparator(OpBuilder &builder, Location loc,
                       Region &comparator, Value lhs, Value rhs) {
  Block &srcBlock = comparator.front();
  IRMapping mapping;
  mapping.map(srcBlock.getArgument(0), lhs);
  mapping.map(srcBlock.getArgument(1), rhs);

  for (Operation &op : srcBlock.without_terminator()) {
    builder.clone(op, mapping);
  }

  auto yieldOp = cast<YieldOp>(srcBlock.getTerminator());
  return mapping.lookup(yieldOp.getOperand(0));
}
```

---

## KernelConfig.cpp Configuration

To route `ArgCompareOp` through the VectorDistribute pipeline, we need to add configuration in [KernelConfig.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp).

### Option 1: Add setArgCompareReductionConfig (Recommended)

Following the pattern of `setReductionConfig` in [ReductionConfigUtils.cpp](compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ReductionConfigUtils.cpp):

```cpp
// In ReductionConfigUtils.cpp or new ArgCompareConfigUtils.cpp

LogicalResult setArgCompareReductionConfig(IREE::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           IREE::LinalgExt::ArgCompareOp op) {
  MLIRContext *context = op.getContext();
  OpBuilder b(context);

  // Get reduction dimension info
  int64_t reductionDim = op.getDimension();
  auto inputType = cast<ShapedType>(op.getInput().getType());
  ArrayRef<int64_t> bounds = inputType.getShape();
  int64_t reductionSize = bounds[reductionDim];

  // Get target info
  IREE::GPU::TargetWgpAttr wgp = target.getWgp();
  int64_t subgroupSize = target.getPreferredSubgroupSize();

  // Calculate workgroup configuration
  // ... (similar to setReductionConfig)

  // Build lowering config with nested layout attributes
  SmallVector<int64_t> threadCounts(numLoops, 1);
  threadCounts[reductionDim] = subgroupSize;

  // Set translation info to VectorDistribute pipeline
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context, CodeGenPipeline::LLVMGPUVectorDistribute, SymbolRefAttr(),
      {workgroupSize, 1, 1}, subgroupSize, pipelineConfig);

  return setTranslationInfo(entryPoint, translationInfo);
}
```

### Option 2: Add to setRootConfig

Add `ArgCompareOp` handling in `setRootConfig` (around line 2291):

```cpp
// After LinalgOp handling:
if (auto argCompareOp = dyn_cast<IREE::LinalgExt::ArgCompareOp>(computeOp)) {
  if (clGPUEnableReductionVectorDistribution) {
    if (succeeded(IREE::GPU::setArgCompareReductionConfig(
            target, entryPointFn, argCompareOp))) {
      LDBG() << "Vector Distribution ArgCompare Reduction Config";
      return success();
    }
  }
}
```

### Configuration Flow

```
setRootConfig (KernelConfig.cpp)
    ↓
detectArgCompareOp
    ↓
clGPUEnableReductionVectorDistribution flag check
    ↓
setArgCompareReductionConfig
    ↓
TranslationInfo: LLVMGPUVectorDistribute
```

---

## Implementation Plan

### Phase 1: Infrastructure (Weeks 1-2)

1. **Add `amdgpu.subgroup_arg_compare` operation** to AMDGPU dialect
   - Location: Create in IREE's AMDGPU extensions or propose upstream
   - Add verifier, printer/parser
   - Add `isArgMax()`, `isArgMin()`, `isCustomComparator()` helpers

2. **Add DPP lowering for `amdgpu.subgroup_arg_compare`**
   - Location: New pattern in AMDGPU → ROCDL lowering
   - Implement comparator region inlining
   - Handle chipset differences (gfx9 vs gfx10+)

### Phase 2: KernelConfig Integration (Week 2)

3. **Add `setArgCompareReductionConfig`**
   - Location: [ReductionConfigUtils.cpp](compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ReductionConfigUtils.cpp)
   - Follow pattern of `setReductionConfig`

4. **Add ArgCompareOp dispatch in setRootConfig**
   - Location: [KernelConfig.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp)
   - Gate with `clGPUEnableReductionVectorDistribution` flag

### Phase 3: Vectorization (Weeks 3-4)

5. **Add ArgCompareOp handling to GenericVectorizationPass**
   - Location: [GenericVectorization.cpp](compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp)
   - Add `IREE::LinalgExt::ArgCompareOp` to candidates list (similar to GatherOp handling)
   - Create custom vectorization function `vectorizeArgCompareOp()`:
     - Read input as vector via `vector.transfer_read`
     - Initialize index vector with element indices
     - Generate vectorized reduction with paired (value, index) outputs
   - Alternative: Add pattern to [VectorizeIREELinalgExtOps.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/VectorizeIREELinalgExtOps.cpp) and call from GenericVectorization

### Phase 4: Layout Configuration and Analysis (Week 4-5)

6. **Add ToLayoutOp anchor insertion for ArgCompareOp**
   - Location: [LLVMGPUConfigureTensorLayouts.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp)
   - Add `setArgCompareAnchor()` function similar to `setContractionAnchor()`
   - Insert `iree_vector_ext.to_layout` ops on ArgCompareOp's vectorized outputs
   - Configure `NestedLayoutAttr` with appropriate `thread_tile` for reduction dimension
   - Ensure both value and index vectors get **identical** layouts (they must be distributed the same way)

7. **Add layout propagation for ArgCompareOp in VectorLayoutAnalysis**
   - Location: [VectorLayoutAnalysis.cpp](compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp)
   - Add forward propagation: when ArgCompareOp source has a layout, project it for the result (similar to `MultiDimReductionOp` handling at lines 108-126)
   - Add backward propagation: propagate layout from results back to accumulator operands (similar to lines 218-221)
   - Key difference from `MultiDimReductionOp`: ArgCompareOp has **two results** (value + index), both need the same projected layout

### Phase 5: Distribution (Week 5-6)

8. **Add `DistributeArgCompare` pattern**
   - Location: [GPUNestedLayoutDistributionPatterns.cpp](compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp)
   - Follow pattern of `DistributeMultiReduction`
   - Generate `amdgpu.subgroup_arg_compare` for cross-thread reduction

### Phase 6: Testing (Week 6-7)

9. **Unit tests**
   - DPP lowering tests
   - Vectorization tests
   - Distribution tests
   - Layout analysis tests

10. **End-to-end tests**
   - Extend [tests/e2e/linalg_ext_ops/arg_compare.mlir](tests/e2e/linalg_ext_ops/arg_compare.mlir)
   - Add GPU-specific tests with various reduction sizes

---

## Key Files Reference

| Component | File | Lines/Purpose |
|-----------|------|---------------|
| **ArgCompareOp Definition** | [LinalgExtOps.td](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td) | 645-770 |
| **ArgCompareOp Verification** | [LinalgExtOps.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp) | 1154-1285 |
| **TilingInterface Impl** | [TilingInterfaceImpl.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp) | 1333-1698 |
| **GenericVectorizationPass** | [GenericVectorization.cpp](compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp) | Main vectorization pass - add ArgCompareOp handling |
| **LinalgExt Vectorization** | [VectorizeIREELinalgExtOps.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/VectorizeIREELinalgExtOps.cpp) | Custom vectorization patterns for LinalgExt ops |
| **Layout Configuration** | [LLVMGPUConfigureTensorLayouts.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp) | ToLayoutOp insertion - add setArgCompareAnchor() |
| **Layout Analysis** | [VectorLayoutAnalysis.cpp](compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp) | Layout propagation - add ArgCompareOp handling |
| **Distribution Patterns** | [GPUNestedLayoutDistributionPatterns.cpp](compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp) | DistributeMultiReduction reference (966-1395) |
| **KernelConfig** | [KernelConfig.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp) | setRootConfig, add ArgCompareOp dispatch |
| **ReductionConfigUtils** | [ReductionConfigUtils.cpp](compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ReductionConfigUtils.cpp) | Add setArgCompareReductionConfig |
| **UKernel ArgMax (reference)** | [iree_uk_amdgpu_argmax_f32i64.c](compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c) | Algorithm reference |
| **isArgmaxOp utility** | [Utils.cpp](compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp) | 904-959 |
| **DPP Lowering Reference** | [SubgroupReduceLowering.cpp](third_party/llvm-project/mlir/lib/Dialect/GPU/Transforms/SubgroupReduceLowering.cpp) | createSubgroupDPPReduction |
| **E2E Tests** | [arg_compare.mlir](tests/e2e/linalg_ext_ops/arg_compare.mlir) | Extend for GPU |

---

## IR Transformation Examples (End-to-End)

This section shows the complete IR transformation at each stage of the pipeline for a simple argmax operation reducing `tensor<4x64xf32>` along dimension 1.

**Configuration**: Workgroup size `[64, 1, 1]`, Subgroup size `64`, 4 parallel batches

---

### Stage 1: Input IR (After Dispatch Formation)

```mlir
// Entry point function with translation info set by KernelConfig
// TranslationInfo: LLVMGPUVectorDistribute, workgroup=[64,1,1], subgroup=64
func.func @argmax_2d_dispatch() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c0_f32 = arith.constant -3.40282e+38 : f32  // -FLT_MAX (identity for max)
  %c0_i32 = arith.constant 0 : i32

  %input = hal.interface.binding.subspan ... : !flow.dispatch.tensor<readonly:tensor<4x64xf32>>
  %output_val = hal.interface.binding.subspan ... : !flow.dispatch.tensor<writeonly:tensor<4xf32>>
  %output_idx = hal.interface.binding.subspan ... : !flow.dispatch.tensor<writeonly:tensor<4xi32>>

  %input_tensor = flow.dispatch.tensor.load %input ... : tensor<4x64xf32>
  %init_val = tensor.empty() : tensor<4xf32>
  %init_idx = tensor.empty() : tensor<4xi32>
  %fill_val = linalg.fill ins(%c0_f32 : f32) outs(%init_val : tensor<4xf32>) -> tensor<4xf32>
  %fill_idx = linalg.fill ins(%c0_i32 : i32) outs(%init_idx : tensor<4xi32>) -> tensor<4xi32>

  %result:2 = iree_linalg_ext.arg_compare dimension(1)
    ins(%input_tensor : tensor<4x64xf32>)
    outs(%fill_val, %fill_idx : tensor<4xf32>, tensor<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>

  flow.dispatch.tensor.store %result#0, %output_val ...
  flow.dispatch.tensor.store %result#1, %output_idx ...
  return
}
```

---

### Stage 2: After Tiling (TileAndDistributeToWorkgroups)

```mlir
// Workgroup processes 1 batch row (4 batches = 4 workgroups in y dimension)
// Each workgroup has 64 threads reducing 64 elements
func.func @argmax_2d_dispatch() {
  %workgroup_id_y = hal.interface.workgroup.id[1] : index  // batch index 0-3

  // Load one row of input for this workgroup
  %input_slice = tensor.extract_slice %input_tensor[%workgroup_id_y, 0] [1, 64] [1, 1]
      : tensor<4x64xf32> to tensor<1x64xf32>

  %init_val_slice = tensor.empty() : tensor<1xf32>
  %init_idx_slice = tensor.empty() : tensor<1xi32>
  %fill_val = linalg.fill ins(%c0_f32) outs(%init_val_slice) -> tensor<1xf32>
  %fill_idx = linalg.fill ins(%c0_i32) outs(%init_idx_slice) -> tensor<1xi32>

  // Arg compare on the slice (1x64 -> 1)
  %result:2 = iree_linalg_ext.arg_compare dimension(1)
    ins(%input_slice : tensor<1x64xf32>)
    outs(%fill_val, %fill_idx : tensor<1xf32>, tensor<1xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<1xf32>, tensor<1xi32>

  // Store results
  tensor.insert_slice %result#0 into %output_val[%workgroup_id_y] [1] [1] ...
  tensor.insert_slice %result#1 into %output_idx[%workgroup_id_y] [1] [1] ...
  return
}
```

---

### Stage 3: After Vectorization (GenericVectorizationPass)

```mlir
// ArgCompareOp is vectorized to read full vectors and perform vector arg_compare
func.func @argmax_2d_dispatch() {
  %workgroup_id_y = hal.interface.workgroup.id[1] : index

  // Vectorized read of input row (64 elements as vector)
  %c0 = arith.constant 0 : index
  %pad = arith.constant -3.40282e+38 : f32
  %input_vec = vector.transfer_read %input_tensor[%workgroup_id_y, %c0], %pad
      {in_bounds = [true, true]} : tensor<4x64xf32>, vector<1x64xf32>

  // Collapse the leading unit dimension
  %input_1d = vector.shape_cast %input_vec : vector<1x64xf32> to vector<64xf32>

  // Initialize index vector with element indices [0, 1, 2, ..., 63]
  %indices = vector.step : vector<64xi32>

  // Initialize accumulators
  %acc_val = arith.constant dense<-3.40282e+38> : vector<1xf32>
  %acc_idx = arith.constant dense<0> : vector<1xi32>

  // Vectorized arg_compare: reduces vector<64xf32> -> f32 (with index)
  // This is a new vector operation that needs to be defined
  %result_val, %result_idx = vector.arg_compare <ogt>, %input_1d, %indices, %acc_val, %acc_idx
      {reduction_dims = [0]} : vector<64xf32>, vector<64xi32> -> f32, i32

  // Write scalar results
  %result_val_vec = vector.broadcast %result_val : f32 to vector<1xf32>
  %result_idx_vec = vector.broadcast %result_idx : i32 to vector<1xi32>
  vector.transfer_write %result_val_vec, %output_val[%workgroup_id_y] ...
  vector.transfer_write %result_idx_vec, %output_idx[%workgroup_id_y] ...
  return
}
```

---

### Stage 4: After Layout Configuration (LLVMGPUConfigureTensorLayouts)

```mlir
// ToLayoutOp anchors inserted to specify how vectors are distributed across threads
// NestedLayoutAttr: thread_tile=[64] means 64 threads each get 1 element
func.func @argmax_2d_dispatch() {
  %workgroup_id_y = hal.interface.workgroup.id[1] : index

  %input_vec = vector.transfer_read %input_tensor[%workgroup_id_y, %c0], %pad
      : tensor<4x64xf32>, vector<64xf32>

  // Layout anchor on input: distribute 64 elements across 64 threads
  %input_layout = iree_vector_ext.to_layout %input_vec
      to layout(#iree_vector_ext.nested_layout<
          subgroup_tile = [1],
          batch_tile = [1],
          outer_tile = [1],
          thread_tile = [64],
          element_tile = [1],
          subgroup_strides = [0],
          thread_strides = [1]
      >) : vector<64xf32>

  // Index vector with same layout
  %indices = vector.step : vector<64xi32>
  %indices_layout = iree_vector_ext.to_layout %indices
      to layout(#iree_vector_ext.nested_layout<
          subgroup_tile = [1],
          batch_tile = [1],
          outer_tile = [1],
          thread_tile = [64],
          element_tile = [1],
          subgroup_strides = [0],
          thread_strides = [1]
      >) : vector<64xi32>

  // Vectorized arg_compare with layout-annotated inputs
  %result_val, %result_idx = vector.arg_compare <ogt>, %input_layout, %indices_layout
      {reduction_dims = [0]} : vector<64xf32>, vector<64xi32> -> f32, i32

  // Result layout anchor (scalar result, no distribution needed)
  %result_val_layout = iree_vector_ext.to_layout %result_val
      to layout(#iree_vector_ext.nested_layout<
          subgroup_tile = [],
          batch_tile = [],
          outer_tile = [],
          thread_tile = [],
          element_tile = [],
          subgroup_strides = [],
          thread_strides = []
      >) : f32

  %result_idx_layout = iree_vector_ext.to_layout %result_idx
      to layout(#iree_vector_ext.nested_layout<...>) : i32

  // Write results
  vector.transfer_write %result_val_layout, %output_val[%workgroup_id_y] ...
  vector.transfer_write %result_idx_layout, %output_idx[%workgroup_id_y] ...
  return
}
```

---

### Stage 5: After Layout Analysis (VectorLayoutAnalysis propagation)

```mlir
// Layout information propagated to all vector values
// This is an analysis pass - IR structure unchanged, but layout map populated
// The analysis determines:
//   - %input_vec: nested_layout<thread_tile=[64]>
//   - %indices: nested_layout<thread_tile=[64]>
//   - %result_val: scalar (projected layout after reduction)
//   - %result_idx: scalar (projected layout after reduction)
//
// Forward propagation: source layout -> projected result layout
// Backward propagation: result layout -> accumulator layout
//
// For ArgCompareOp specifically:
//   - Input layout with thread_tile=[64] on reduction dim
//   - Result layout = input_layout.project([true]) -> scalar
//   - Both value and index results get identical projected layouts
```

---

### Stage 6: After Distribution (LLVMGPUVectorDistribute - DistributeArgCompare)

```mlir
// Vectors distributed across threads based on NestedLayoutAttr
// Each thread reads its portion and performs local + cross-thread reduction
func.func @argmax_2d_dispatch() {
  %thread_id_x = gpu.thread_id x  // 0-63
  %workgroup_id_y = hal.interface.workgroup.id[1] : index

  // Compute this thread's read index
  // With thread_tile=[64], thread N reads element N
  %read_idx = affine.apply affine_map<(d0) -> (d0)>(%thread_id_x)

  // Each thread reads 1 element (distributed read)
  %thread_val = vector.transfer_read %input_tensor[%workgroup_id_y, %read_idx], %pad
      : tensor<4x64xf32>, vector<1xf32>

  // Each thread's index is its thread_id
  %thread_idx_i32 = arith.index_cast %thread_id_x : index to i32
  %thread_idx = vector.broadcast %thread_idx_i32 : i32 to vector<1xi32>

  // Reshape to distributed shape: batch x outer x element = 1x1x1
  %dis_val = vector.shape_cast %thread_val : vector<1xf32> to vector<1x1x1xf32>
  %dis_idx = vector.shape_cast %thread_idx : vector<1xi32> to vector<1x1x1xi32>

  // Local thread reduction (trivial for 1 element - just extract)
  %local_val = vector.extract %dis_val[0, 0, 0] : f32 from vector<1x1x1xf32>
  %local_idx = vector.extract %dis_idx[0, 0, 0] : i32 from vector<1x1x1xi32>

  // Cross-thread reduction via amdgpu.subgroup_arg_compare
  // This replaces the vector.arg_compare for thread_tile > 1
  %reduced_val, %reduced_idx = amdgpu.subgroup_arg_compare %local_val, %local_idx
      cluster(size = 64) : (f32, i32) -> (f32, i32) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      amdgpu.yield %cmp : i1
  }

  // Only thread 0 writes the final result
  %c0 = arith.constant 0 : index
  %is_thread_0 = arith.cmpi eq, %thread_id_x, %c0 : index
  scf.if %is_thread_0 {
    %result_val_vec = vector.broadcast %reduced_val : f32 to vector<1xf32>
    %result_idx_vec = vector.broadcast %reduced_idx : i32 to vector<1xi32>
    vector.transfer_write %result_val_vec, %output_val[%workgroup_id_y]
        : vector<1xf32>, tensor<4xf32>
    vector.transfer_write %result_idx_vec, %output_idx[%workgroup_id_y]
        : vector<1xi32>, tensor<4xi32>
  }
  return
}
```

---

### Stage 7: After AMDGPU Lowering (amdgpu.subgroup_arg_compare -> DPP + Ballot)

```mlir
// amdgpu.subgroup_arg_compare lowered to DPP butterfly reduction
func.func @argmax_2d_dispatch() {
  %thread_id_x = gpu.thread_id x
  %workgroup_id_y = hal.interface.workgroup.id[1] : index

  // ... (same as before until cross-thread reduction) ...

  %local_val = ...
  %local_idx = ...

  // ======== DPP Butterfly Reduction (6 stages for 64 threads) ========

  // Stage 1: Reduce pairs (lanes N <-> N XOR 1)
  // quad_perm([1,0,3,2]) swaps lanes 0<->1, 2<->3, etc.
  %dpp_val_1 = amdgpu.dpp %local_val %local_val quad_perm([1,0,3,2])
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : f32
  %dpp_idx_1 = amdgpu.dpp %local_idx %local_idx quad_perm([1,0,3,2])
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : i32
  // Inline comparator: is local_val > dpp_val_1?
  %cmp_1 = arith.cmpf ogt, %local_val, %dpp_val_1 : f32
  %val_1 = arith.select %cmp_1, %local_val, %dpp_val_1 : f32
  %idx_1 = arith.select %cmp_1, %local_idx, %dpp_idx_1 : i32
  // Tie-breaking: if values equal, pick smaller index
  %eq_1 = arith.cmpf oeq, %local_val, %dpp_val_1 : f32
  %smaller_idx_1 = arith.minsi %local_idx, %dpp_idx_1 : i32
  %idx_1_final = arith.select %eq_1, %smaller_idx_1, %idx_1 : i32

  // Stage 2: Reduce quads (lanes N <-> N XOR 2)
  // quad_perm([2,3,0,1]) swaps lanes 0<->2, 1<->3, etc.
  %dpp_val_2 = amdgpu.dpp %val_1 %val_1 quad_perm([2,3,0,1])
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : f32
  %dpp_idx_2 = amdgpu.dpp %idx_1_final %idx_1_final quad_perm([2,3,0,1])
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : i32
  %cmp_2 = arith.cmpf ogt, %val_1, %dpp_val_2 : f32
  %val_2 = arith.select %cmp_2, %val_1, %dpp_val_2 : f32
  %idx_2 = arith.select %cmp_2, %idx_1_final, %dpp_idx_2 : i32
  %eq_2 = arith.cmpf oeq, %val_1, %dpp_val_2 : f32
  %smaller_idx_2 = arith.minsi %idx_1_final, %dpp_idx_2 : i32
  %idx_2_final = arith.select %eq_2, %smaller_idx_2, %idx_2 : i32

  // Stage 3: Reduce 8-lane groups (lanes N <-> N XOR 4)
  // row_half_mirror swaps lanes 0<->4, 1<->5, 2<->6, 3<->7 within each row
  %dpp_val_3 = amdgpu.dpp %val_2 %val_2 row_half_mirror
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : f32
  %dpp_idx_3 = amdgpu.dpp %idx_2_final %idx_2_final row_half_mirror
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : i32
  %cmp_3 = arith.cmpf ogt, %val_2, %dpp_val_3 : f32
  %val_3 = arith.select %cmp_3, %val_2, %dpp_val_3 : f32
  %idx_3 = arith.select %cmp_3, %idx_2_final, %dpp_idx_3 : i32
  %eq_3 = arith.cmpf oeq, %val_2, %dpp_val_3 : f32
  %smaller_idx_3 = arith.minsi %idx_2_final, %dpp_idx_3 : i32
  %idx_3_final = arith.select %eq_3, %smaller_idx_3, %idx_3 : i32

  // Stage 4: Reduce 16-lane groups (lanes N <-> N XOR 8)
  // row_mirror swaps lanes 0<->8, 1<->9, ..., 7<->15 within each row
  %dpp_val_4 = amdgpu.dpp %val_3 %val_3 row_mirror
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : f32
  %dpp_idx_4 = amdgpu.dpp %idx_3_final %idx_3_final row_mirror
      {row_mask = 0xf : i32, bank_mask = 0xf : i32, bound_ctrl = true} : i32
  %cmp_4 = arith.cmpf ogt, %val_3, %dpp_val_4 : f32
  %val_4 = arith.select %cmp_4, %val_3, %dpp_val_4 : f32
  %idx_4 = arith.select %cmp_4, %idx_3_final, %dpp_idx_4 : i32
  %eq_4 = arith.cmpf oeq, %val_3, %dpp_val_4 : f32
  %smaller_idx_4 = arith.minsi %idx_3_final, %dpp_idx_4 : i32
  %idx_4_final = arith.select %eq_4, %smaller_idx_4, %idx_4 : i32

  // Stage 5: Reduce across rows (32-lane, lanes N <-> N XOR 16)
  // For gfx10+: use permlanex16
  %dpp_val_5 = rocdl.permlanex16 %val_4, %val_4, %c_1, %c_1
      {fi = true, bc = true} : (f32, f32, i32, i32) -> f32
  %dpp_idx_5 = rocdl.permlanex16 %idx_4_final, %idx_4_final, %c_1, %c_1
      {fi = true, bc = true} : (i32, i32, i32, i32) -> i32
  %cmp_5 = arith.cmpf ogt, %val_4, %dpp_val_5 : f32
  %val_5 = arith.select %cmp_5, %val_4, %dpp_val_5 : f32
  %idx_5 = arith.select %cmp_5, %idx_4_final, %dpp_idx_5 : i32
  %eq_5 = arith.cmpf oeq, %val_4, %dpp_val_5 : f32
  %smaller_idx_5 = arith.minsi %idx_4_final, %dpp_idx_5 : i32
  %idx_5_final = arith.select %eq_5, %smaller_idx_5, %idx_5 : i32

  // Stage 6: Final reduction (64-lane, lanes N <-> N XOR 32)
  // Read from lane 32 and combine with lane 0's value
  %c32 = arith.constant 32 : i32
  %val_from_32 = rocdl.readlane %val_5, %c32 : (f32, i32) -> f32
  %idx_from_32 = rocdl.readlane %idx_5_final, %c32 : (i32, i32) -> i32
  %cmp_6 = arith.cmpf ogt, %val_5, %val_from_32 : f32
  %val_6 = arith.select %cmp_6, %val_5, %val_from_32 : f32
  %idx_6 = arith.select %cmp_6, %idx_5_final, %idx_from_32 : i32
  %eq_6 = arith.cmpf oeq, %val_5, %val_from_32 : f32
  %smaller_idx_6 = arith.minsi %idx_5_final, %idx_from_32 : i32
  %idx_6_final = arith.select %eq_6, %smaller_idx_6, %idx_6 : i32

  // ======== Final Ballot-Based Winner Selection ========
  // Find which lanes have the winning value (handles ties)
  %c0_i32 = arith.constant 0 : i32
  %final_max = rocdl.readlane %val_6, %c0_i32 : (f32, i32) -> f32
  %has_max = arith.cmpf oeq, %val_6, %final_max : f32
  %ballot_mask = rocdl.ballot %has_max : (i1) -> i64

  // Find the first lane with the max (smallest index due to tie-breaking)
  %winner_lane_i64 = llvm.intr.cttz(%ballot_mask, %false) : (i64, i1) -> i64
  %winner_lane = arith.trunci %winner_lane_i64 : i64 to i32

  // Read the winning index from that lane
  %final_idx = rocdl.readlane %idx_6_final, %winner_lane : (i32, i32) -> i32

  // Only thread 0 writes
  %c0 = arith.constant 0 : index
  %is_thread_0 = arith.cmpi eq, %thread_id_x, %c0 : index
  scf.if %is_thread_0 {
    vector.transfer_write ... %final_max ...
    vector.transfer_write ... %final_idx ...
  }
  return
}
```

---

### Stage 8: After ROCDL to LLVM Lowering

```mlir
// Final LLVM IR with AMD GCN intrinsics
llvm.func @argmax_2d_dispatch() {
  %thread_id = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32
  %workgroup_id = llvm.call @llvm.amdgcn.workgroup.id.y() : () -> i32

  // ... load operations ...

  // DPP operations become llvm.amdgcn.update.dpp
  %dpp_val_1 = llvm.call @llvm.amdgcn.update.dpp.f32(
      %local_val, %local_val,
      %c312,      // quad_perm([1,0,3,2]) = 0b01001110 = 78... encoded as DPP ctrl
      %c15,       // row_mask
      %c15,       // bank_mask
      %true       // bound_ctrl
  ) : (f32, f32, i32, i32, i32, i1) -> f32

  // ... more DPP stages ...

  // Ballot becomes llvm.amdgcn.ballot
  %ballot = llvm.call @llvm.amdgcn.ballot.i64(%has_max) : (i1) -> i64

  // Count trailing zeros
  %winner = llvm.call @llvm.cttz.i64(%ballot, %false) : (i64, i1) -> i64

  // Readlane becomes llvm.amdgcn.readlane
  %final_idx = llvm.call @llvm.amdgcn.readlane.i32(%idx_6, %winner_i32) : (i32, i32) -> i32

  // Conditional store
  llvm.cond_br %is_thread_0, ^bb_store, ^bb_exit

^bb_store:
  llvm.store %final_max, %output_val_ptr : f32, !llvm.ptr
  llvm.store %final_idx, %output_idx_ptr : i32, !llvm.ptr
  llvm.br ^bb_exit

^bb_exit:
  llvm.return
}
```

---

### Summary: End-to-End Pipeline Stages

| Stage | Pass/Transform | Key IR Changes |
|-------|----------------|----------------|
| 1 | Input | `iree_linalg_ext.arg_compare` on tensor |
| 2 | TileAndDistribute | Tiled to workgroup-local tensors |
| 3 | GenericVectorization | `vector.arg_compare` on vectors |
| 4 | LLVMGPUConfigureTensorLayouts | `iree_vector_ext.to_layout` anchors inserted |
| 5 | VectorLayoutAnalysis | Layouts propagated to all vector values |
| 6 | LLVMGPUVectorDistribute | Distributed reads + `amdgpu.subgroup_arg_compare` |
| 7 | AMDGPU Lowering | DPP butterfly + ballot + readlane |
| 8 | ROCDL to LLVM | `llvm.amdgcn.*` intrinsics |

---

## Open Questions

1. **Should `amdgpu.subgroup_arg_compare` be IREE-specific or upstream MLIR?**
   - **Recommendation**: Start as IREE-specific, propose upstream once design stabilizes

2. **Comparator region complexity limits**
   - Should we limit what operations are allowed in comparator regions?
   - **Recommendation**: Start with pure arithmetic/math ops only

3. **Performance trade-off: Region inlining vs. predefined patterns**
   - Inlining custom comparators at each DPP stage increases code size
   - **Recommendation**: Use hybrid approach - detect standard patterns for optimization

4. **Integration with existing ukernel path**
   - Should VectorDistribute replace the ukernel path or complement it?
   - **Recommendation**: Complement - ukernels for `linalg.generic` argmax, VectorDistribute for `iree_linalg_ext.arg_compare`

5. **Tie-breaking semantics**
   - Current design uses smallest index for ties
   - **Recommendation**: Keep as default, consider optional attribute later

---

## Conclusion

Supporting `ArgCompareOp` in the VectorDistribute pipeline requires:

1. **New AMDGPU dialect operation**: `amdgpu.subgroup_arg_compare` that carries the comparator region and leverages DPP + ballot

2. **Comparator analysis**: Helper methods to detect standard argmax/argmin patterns for optimized lowering

3. **Region inlining infrastructure**: Clone comparator regions at each DPP reduction stage

4. **Vectorization patterns**: Dual-output vectorization maintaining value-index pairs

5. **Layout configuration**: Insert `ToLayoutOp` anchors for ArgCompareOp via `setArgCompareAnchor()` in [LLVMGPUConfigureTensorLayouts.cpp](compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp)

6. **Layout analysis**: Add ArgCompareOp handling in [VectorLayoutAnalysis.cpp](compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp) for forward/backward layout propagation

7. **Distribution patterns**: `DistributeArgCompare` pattern generating `amdgpu.subgroup_arg_compare`

8. **KernelConfig integration**: Route `ArgCompareOp` to VectorDistribute pipeline

**Pipeline Summary**:
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
amdgpu.subgroup_arg_compare (with comparator region)
    ↓ (AMDGPUToROCDL)
amdgpu.dpp + rocdl.ballot + rocdl.readlane
    ↓ (ROCDLToLLVM)
llvm.amdgcn.update.dpp + llvm.amdgcn.ballot + llvm.amdgcn.readlane
```

**Recommended Approach**: **Hybrid (Option C)** - Use `amdgpu.subgroup_arg_compare` with built-in comparator analysis to detect standard argmax/argmin patterns for optimized lowering, falling back to general region inlining for custom comparators.
