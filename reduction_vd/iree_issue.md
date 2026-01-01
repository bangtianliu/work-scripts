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

### Current VectorDistribute Pipeline

The VectorDistribute pipeline (documented in your DPP handling document) follows:

```
linalg.generic → vector.multi_reduction → gpu.subgroup_reduce → amdgpu.dpp → rocdl.update.dpp
```

For `arg_compare`, the proposed pipeline is:

```
iree_linalg_ext.arg_compare
    ↓ (KernelConfig: LLVMGPUVectorDistribute)
    ↓ (GenericVectorizationPass)
vectorized arg_compare (value + index vectors)
    ↓ (LLVMGPUConfigureTensorLayouts)
iree_vector_ext.to_layout with NestedLayoutAttr
    ↓ (VectorLayoutAnalysis)
all vector values have layouts
    ↓ (LLVMGPUVectorDistribute)
amdgpu.subgroup_arg_compare (with comparator region)
    ↓ (AMDGPUToROCDL)
amdgpu.dpp + rocdl.ballot + rocdl.readlane
```

Key pattern: `DistributeMultiReduction` in [GPUNestedLayoutDistributionPatterns.cpp:966-1395](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp#L966-L1395)

### Existing ArgMax Implementation (ROCM UKernel)

The ROCM argmax ukernel in [iree_uk_amdgpu_argmax_f32i64.c](https://github.com/iree-org/iree/blob/main/compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c) demonstrates the key algorithm:

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

## Proposed Solution

### High-Level Approach

1. Route `ArgCompareOp` through VectorDistribute pipeline via `KernelConfig.cpp`
2. Add vectorization pattern for `ArgCompareOp`
3. Add layout configuration and analysis for `ArgCompareOp`
4. Create new `amdgpu.subgroup_arg_compare` operation for AMD-specific lowering
5. Lower to DPP operations + ballot for efficient execution

---

## TODO Plan

- [ ] **KernelConfig Integration**
  Add `setArgCompareReductionConfig` in `ReductionConfigUtils.cpp` following the same pattern as `setReductionConfig`. Add ArgCompareOp case in `KernelConfig.cpp:setRootConfig`, gated with `clGPUEnableReductionVectorDistribution` flag.

- [ ] **AMDGPU Dialect Operation**
  Define `amdgpu.subgroup_arg_compare` operation with comparator region, value/index inputs, and dual outputs. Add helper methods `isArgMax()`, `isArgMin()`, `isCustomComparator()` for pattern detection. Include verifier, printer, and parser.

- [ ] **Vectorization**
  Add ArgCompareOp handling to `GenericVectorizationPass` in `GenericVectorization.cpp`. Handle dual outputs (value + index vectors). Alternative: Add pattern to `VectorizeIREELinalgExtOps.cpp`.

- [ ] **Layout Configuration**
  Add `setArgCompareAnchor()` function in `LLVMGPUConfigureTensorLayouts.cpp` similar to `setContractionAnchor()`. Insert `iree_vector_ext.to_layout` ops on ArgCompareOp's vectorized outputs. Ensure both value and index vectors get **identical** layouts.

- [ ] **Layout Analysis**
  Add ArgCompareOp handling in `VectorLayoutAnalysis.cpp`. Add forward propagation (project source layout for result, similar to `MultiDimReductionOp` at lines 108-126). Add backward propagation (propagate layout from results to accumulators, similar to lines 218-221). Both value and index results need the same projected layout.

- [ ] **Distribution**
  Add `DistributeArgCompare` pattern in `GPUNestedLayoutDistributionPatterns.cpp`. Handle local reduction for thread-local elements, then create `amdgpu.subgroup_arg_compare` for subgroup-level reduction with value-index pairs.

- [ ] **DPP Lowering**
  Add `SubgroupArgCompareToROCDL` pattern. Implement optimized paths for argmax/argmin using `arith.maximumf`/`minimumf`. Implement general path with region inlining for custom comparators (clone at each DPP stage). Handle chipset differences (gfx9: row_bcast, gfx10+: permlanex16). Add ballot-based winner selection with tie-breaking (smallest index wins).

- [ ] **Testing**
  Add unit tests for config selection, vectorization, layout analysis, distribution, and DPP lowering. Add E2E tests for argmax, argmin, and custom comparator cases.

## Key Files
- `KernelConfig.cpp`, `ReductionConfigUtils.cpp` - config
- `GenericVectorization.cpp`, `VectorizeIREELinalgExtOps.cpp` - vectorization
- `LLVMGPUConfigureTensorLayouts.cpp` - layout configuration
- `VectorLayoutAnalysis.cpp` - layout analysis
- `GPUNestedLayoutDistributionPatterns.cpp` - distribution
- AMDGPU → ROCDL lowering - DPP lowering

## Reference
- Design: `ArgCompareOp_VectorDistribute_Design.md`
- UKernel: `compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c`
