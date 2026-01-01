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
| After GenericVectorizationPass | `vector.multi_reduction <add>, %14, %cst_1 [0] : vector<64xf32> to f32` |
| After layout annotation | `iree_vector_ext.to_layout ... nested_layout<thread_tile=[64]>` |
| After LLVMGPUVectorDistributePass | Each thread reads `vector<1xf32>`, local reduction to `f32`, then `gpu.subgroup_reduce add cluster(size=64)` |

---

## Design Decisions

1. **Butterfly Shuffle Pattern**: Reduces communication overhead by halving stride each iteration (log2(N) steps for N threads)

2. **Nested Layout Abstraction**: Provides flexible distribution specification across subgroups/threads/elements

3. **Shared Memory for Cross-Subgroup**: When reduction spans multiple subgroups, partial results are exchanged via workgroup memory with barrier synchronization

4. **Conditional Write**: Only one thread (typically thread 0) writes the final result to avoid race conditions

5. **Identity Value Awareness**: Different reduction types use appropriate neutral elements for correctness
