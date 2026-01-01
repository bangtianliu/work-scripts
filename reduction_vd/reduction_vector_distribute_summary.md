# Reduction Support in IREE's Vector Distribute Pipeline

## Overview

This document summarizes how reduction operations are handled in IREE's vector distribute pipeline, based on analysis of the codebase and the `check_log.mlir` compilation trace.

## Example from check_log.mlir

The log shows a reduction of shape `8x64xf32 → 8xf32` going through the **LLVMGPUVectorDistribute** pipeline with:
- Workgroup size: `[64, 1, 1]`
- Subgroup size: `64`

---

## Key Transformation Stages

### 1. Before Distribution (Pre-LLVMGPUVectorDistributePass)

```mlir
%6 = iree_vector_ext.to_layout %5 to layout(#iree_vector_ext.nested_layout<
    subgroup_tile = [1], batch_tile = [1], outer_tile = [1],
    thread_tile = [64], element_tile = [1],
    subgroup_strides = [0], thread_strides = [1]>) : vector<64xf32>
%10 = vector.multi_reduction <add>, %9, %cst_0 [0] : vector<64xf32> to f32
```

- The input `vector<64xf32>` has a nested layout with `thread_tile = [64]`
- This means 64 threads will each hold 1 element after distribution

### 2. After LLVMGPUVectorDistributePass

```mlir
// Each thread computes its read index based on thread ID
%8 = affine.linearize_index disjoint [%6#1, %c0, %c0, %7#1, %c0] by (1, 1, 1, 64, 1)

// Each thread reads 1 element
%9 = vector.transfer_read %3[%arg0, %8], %0 : ..., vector<1xf32>

// Insert into thread-local 1x1x1 vector (batch x outer x element)
%10 = vector.insert_strided_slice %9, %cst_0 {offsets = [0, 0, 0]} : vector<1xf32> into vector<1x1x1xf32>

// Local thread reduction (trivial - reduces 1x1x1 to scalar)
%12 = vector.multi_reduction <add>, %11, %cst [0, 1, 2] : vector<1x1x1xf32> to f32

// Cross-thread reduction via subgroup_reduce
%13 = gpu.subgroup_reduce add %12 cluster(size = 64) : (f32) -> f32

// Only thread 0 writes the result
%14 = arith.cmpi eq, %1, %c0 : index
scf.if %14 {
    %15 = vector.broadcast %13 : f32 to vector<f32>
    vector.transfer_write %15, %subview[] : ...
}
```

---

## Nested Layout Structure

The `nested_layout` attribute describes how vectors are distributed across threads:

```
#iree_vector_ext.nested_layout<
    subgroup_tile = [S],      // Distribution across subgroups
    batch_tile = [B],         // Batching within a thread
    outer_tile = [O],         // Outer tiling
    thread_tile = [T],        // Distribution across threads in subgroup
    element_tile = [E],       // Elements per thread
    subgroup_strides = [...], // Stride for subgroup indexing
    thread_strides = [...]    // Stride for thread indexing
>
```

Total vector size = `S × B × O × T × E`

For the example: `1 × 1 × 1 × 64 × 1 = 64` elements distributed across 64 threads.

---

## Reduction Stages in DistributeMultiReduction

The `DistributeMultiReduction` pattern (in `GPUNestedLayoutDistributionPatterns.cpp:966-1395`) performs up to 4 stages:

### Stage 1: Local Thread Reduction
- Each thread reduces its local data across batch, outer, and element tiles
- Creates: `vector.multi_reduction` on the distributed shape

### Stage 2: Thread-Level Cross-Reduction
- Triggered when `thread_tile > 1` on any reduction dimension
- Uses `gpu.subgroup_reduce` with cluster size and stride parameters
- Implements butterfly shuffle pattern for efficiency

### Stage 3: Accumulator Reduction
- Combines the reduction result with the accumulator argument
- Uses appropriate arithmetic operation (e.g., `arith.addf`)

### Stage 4: Subgroup-Level Reduction (if needed)
- Triggered when `subgroup_tile > 1` on any reduction dimension
- Uses shared memory buffering:
  1. Write partial results to workgroup memory
  2. Synchronize with `gpu.barrier`
  3. Read and perform secondary reduction across subgroups

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp` | `DistributeMultiReduction` pattern (lines 966-1395) |
| `compiler/src/iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.cpp` | Core distribution framework and worklist-based pattern application |
| `compiler/src/iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h` | `DistributionPattern` base class definitions |
| `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp` | `warpReduction()` (lines 427-500), `emitGPUGroupReduction()` (lines 589-656) |
| `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUVectorDistribute.cpp` | LLVM GPU specific pass entry point |

---

## Warp Reduction Implementation

The `warpReduction()` function in `GPUUtils.cpp:427-500` implements butterfly shuffle reduction:

```cpp
// Butterfly shuffle pattern
for (uint64_t i = 1; i < numLaneToReduce; i <<= 1) {
    Value shuffled = gpu.shuffle xor laneVal, i, width;
    laneVal = arith.addf laneVal, shuffled;  // or other combining op
}
```

For a 64-thread reduction, this performs 6 iterations with strides: 1, 2, 4, 8, 16, 32.

---

## Supported Reduction Kinds

From `GPUUtils.cpp:562-586`, the following `vector::CombiningKind` operations are supported:

| Category | Operations |
|----------|------------|
| Arithmetic | `ADD`, `MUL` |
| Integer Comparison | `MINUI`, `MINSI`, `MAXUI`, `MAXSI` |
| Float Comparison | `MINUMF`, `MAXNUMF`, `MINIMUMF`, `MAXIMUMF` |
| Bitwise | `AND`, `OR`, `XOR` |

### Identity Values (from `GPUUtils.cpp:504-541`)

| Operation | Identity Value |
|-----------|----------------|
| ADD | 0 |
| MUL | 1 |
| MINUI/MINSI | INT_MAX |
| MAXUI/MAXSI | INT_MIN |
| MINIMUMF/MINNUMF | +∞ |
| MAXIMUMF/MAXNUMF | -∞ |
| AND | all bits set (1) |
| OR, XOR | 0 |

---

## Complete Pipeline Flow

```
Input: vector.multi_reduction <kind>, src, acc [dims]
                    ↓
        GenericVectorizationPass
        (Vectorizes linalg ops)
                    ↓
        Layout annotation with nested_layout
        (iree_vector_ext.to_layout)
                    ↓
        LLVMGPUVectorDistributePass
                    ↓
    ┌───────────────────────────────────────┐
    │ DistributeMultiReduction Pattern      │
    │                                       │
    │ 1. Distribute transfer_read           │
    │    - Compute thread-specific indices  │
    │    - Read thread's portion            │
    │                                       │
    │ 2. Local thread reduction             │
    │    - vector.multi_reduction on local  │
    │                                       │
    │ 3. Cross-thread reduction             │
    │    - gpu.subgroup_reduce              │
    │                                       │
    │ 4. Conditional write (thread 0 only)  │
    └───────────────────────────────────────┘
                    ↓
        Output: Distributed code with
        - gpu.subgroup_reduce for intra-warp
        - Shared memory for inter-subgroup (if needed)
        - Arithmetic ops for combining results
```

---

## Example Trace Summary

From `check_log.mlir`, the transformation for `reduce_dispatch_0_reduction_8x64_f32`:

| Pass | Key IR State |
|------|--------------|
| After LLVMGPUConfigureTensorLayoutsPass | `linalg.reduce` with `iree_vector_ext.to_layout` on `tensor<1x64xf32>` with `thread_tile=[1,64]` |
| After LinalgGeneralizeNamedOpsPass | `linalg.reduce` → `linalg.generic` with `iterator_types = ["reduction"]` |
| After FoldUnitExtentDimsPass | `tensor<1x64xf32>` → `tensor<64xf32>`, layouts updated to 1D |
| After VectorizeIREEVectorExtOpsPass | `iree_vector_ext.to_layout` now on `vector<64xf32>`, reduction still `linalg.generic` |
| After GenericVectorizationPass | `linalg.generic` → `vector.multi_reduction <add>, %14, %cst_1 [0] : vector<64xf32> to f32` |
| After LLVMGPUVectorDistributePass | Each thread reads `vector<1xf32>`, local reduction to `f32`, then `gpu.subgroup_reduce add cluster(size=64)` |

---

## GenericVectorizationPass Details

The `GenericVectorizationPass` (`iree-codegen-generic-vectorization`) is responsible for converting `linalg.generic` operations into their vector equivalents. This pass runs after layout annotations have been attached via `iree_vector_ext.to_layout`.

### What GenericVectorizationPass Does

1. **Converts linalg.generic to vector operations**: Transforms tensor-based linalg ops into vector dialect operations
2. **Handles to_layout ops on tensors**: Vectorizes the `iree_vector_ext.to_layout` operations
3. **Creates vector.multi_reduction**: Converts reduction iterator types into `vector.multi_reduction` operations

### Transformation Example from check_log.mlir

**Before GenericVectorizationPass** (after VectorizeIREEVectorExtOpsPass):
```mlir
// The to_layout ops are already on vectors after VectorizeIREEVectorExtOpsPass
%12 = vector.transfer_read %extracted_slice_1[%c0], %0 {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
%13 = iree_vector_ext.to_layout %12 to layout(#iree_vector_ext.nested_layout<
    subgroup_tile = [1], batch_tile = [1], outer_tile = [1],
    thread_tile = [64], element_tile = [1],
    subgroup_strides = [0], thread_strides = [1]>) : vector<64xf32>

// The reduction is still a linalg.generic
%25 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
} ins(%24 : tensor<64xf32>) outs(%9 : tensor<f32>) {
^bb0(%in: f32, %out: f32):
    %26 = arith.addf %in, %out : f32
    linalg.yield %26 : f32
} -> tensor<f32>
```

**After GenericVectorizationPass**:
```mlir
// All linalg.generic ops are now vectorized
%10 = vector.transfer_read %extracted_slice_2[%c0], %0 {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
%11 = iree_vector_ext.to_layout %10 to layout(#iree_vector_ext.nested_layout<
    subgroup_tile = [1], batch_tile = [1], outer_tile = [1],
    thread_tile = [64], element_tile = [1],
    subgroup_strides = [0], thread_strides = [1]>) : vector<64xf32>

// Element-wise add is now arith.addf on vectors
%13 = arith.addf %11, %12 : vector<64xf32>

%14 = iree_vector_ext.to_layout %13 to layout(...)

// Reduction is now vector.multi_reduction
%15 = vector.multi_reduction <add>, %14, %cst_1 [0] : vector<64xf32> to f32
%16 = vector.broadcast %15 : f32 to vector<f32>
%17 = vector.transfer_write %16, %9[] : vector<f32>, tensor<f32>
```

### Key Observations

1. **Tensor to Vector**: `linalg.generic` ops operating on tensors become vector operations with `vector.transfer_read`/`vector.transfer_write` for I/O

2. **Layout Preservation**: The `iree_vector_ext.to_layout` annotations are preserved and now operate on vector types instead of tensor types

3. **Reduction Conversion**: The reduction `linalg.generic` with `iterator_types = ["reduction"]` becomes `vector.multi_reduction <add>, ..., [0]`

4. **Accumulator Handling**: The initial reduction value (`%cst_1 = -0.000000e+00 : f32`) is passed as the accumulator to `vector.multi_reduction`

### Pipeline Position

```
LLVMGPUConfigureTensorLayoutsPass
    ↓ (adds iree_vector_ext.to_layout on tensors)
LinalgGeneralizeNamedOpsPass
    ↓ (converts linalg.reduce to linalg.generic)
FoldUnitExtentDimsPass
    ↓ (folds unit dims: tensor<1x64xf32> → tensor<64xf32>)
VectorizeIREEVectorExtOpsPass
    ↓ (vectorizes iree_vector_ext.to_layout on tensors)
GenericVectorizationPass  ← HERE
    ↓ (vectorizes remaining linalg.generic ops)
LLVMGPUVectorDistributePass
    ↓ (distributes vectors across threads using layout info)
```

---

## Design Decisions

1. **Butterfly Shuffle Pattern**: Reduces communication overhead by halving stride each iteration (log2(N) steps for N threads)

2. **Nested Layout Abstraction**: Provides flexible distribution specification across subgroups/threads/elements

3. **Shared Memory for Cross-Subgroup**: When reduction spans multiple subgroups, partial results are exchanged via workgroup memory with barrier synchronization

4. **Conditional Write**: Only one thread (typically thread 0) writes the final result to avoid race conditions

5. **Identity Value Awareness**: Different reduction types use appropriate neutral elements for correctness
