# [Feature Request] Support ArgCompareOp in VectorDistribute Pipeline for AMD GPU

## Summary

Add support for `iree_linalg_ext.arg_compare` operations through IREE's VectorDistribute pipeline on AMD GPU backends, leveraging DPP instructions for efficient subgroup-level reductions and `rocdl.ballot` for index computation.

## Motivation

Currently, `ArgCompareOp` (argmax/argmin with index tracking) does not go through the VectorDistribute pipeline on AMD GPUs. This results in suboptimal performance for reduction operations that need to track both the selected value and its index.

The VectorDistribute pipeline provides efficient GPU execution through:
- Vectorized memory access
- Subgroup-level reductions using DPP instructions
- Proper workgroup/thread distribution

## Key Challenges

1. **Dual Output**: `arg_compare` produces two outputs (value and index), unlike standard reductions
2. **User-Defined Comparator Region**: The operation supports arbitrary comparison logic (not just argmax/argmin)
3. **Index Tracking**: Indices must be tracked through DPP-based reductions as data moves between lanes
4. **Tie-Breaking**: When multiple lanes have equal values, the smallest index must be selected

## Proposed Solution

### High-Level Approach

1. Route `ArgCompareOp` through VectorDistribute pipeline via `KernelConfig.cpp`
2. Add vectorization pattern for `ArgCompareOp`
3. Create new `amdgpu.subgroup_arg_compare` operation for AMD-specific lowering
4. Lower to DPP operations + ballot for efficient execution

### Pipeline Flow

```
iree_linalg_ext.arg_compare
    ↓ (KernelConfig: setArgCompareReductionConfig)
LLVMGPUVectorDistribute pipeline
    ↓ (Vectorization)
vectorized arg_compare with NestedLayoutAttr
    ↓ (Distribution)
amdgpu.subgroup_arg_compare (with comparator region)
    ↓ (AMDGPUToROCDL)
amdgpu.dpp + rocdl.ballot + rocdl.readlane
```

---

## TODO Plan

### Phase 1: KernelConfig Integration
- [ ] **1.1** Add `setArgCompareReductionConfig` function in `ReductionConfigUtils.cpp`
  - Follow same pattern as `setReductionConfig`
  - Handle reduction dimension from `op.getDimension()`
  - Calculate workgroup size, subgroup size, thread loads
  - Set `LLVMGPUVectorDistribute` pipeline
- [ ] **1.2** Add header declaration in `ConfigUtils.h`
- [ ] **1.3** Add `ArgCompareOp` dispatch in `KernelConfig.cpp:setRootConfig`
  - Add case after LinalgOp handling (around line 2291)
  - Gate with `clGPUEnableReductionVectorDistribution` flag
- [ ] **1.4** Add unit tests for config selection

### Phase 2: AMDGPU Dialect Operation
- [ ] **2.1** Define `amdgpu.subgroup_arg_compare` operation
  - Support comparator region for custom predicates
  - Accept value and index inputs
  - Return value and index outputs
  - Add `cluster_size` and `cluster_stride` attributes
- [ ] **2.2** Add helper methods: `isArgMax()`, `isArgMin()`, `isCustomComparator()`
- [ ] **2.3** Add verifier, printer/parser
- [ ] **2.4** Add operation tests

### Phase 3: Vectorization
- [ ] **3.1** Add `VectorizeArgCompareOp` pattern in `VectorizeIREELinalgExtOps.cpp`
  - Read input as vector
  - Initialize index vector with lane indices
  - Create vectorized reduction with value-index pairs
  - Write both value and index results
- [ ] **3.2** Add layout propagation for arg_compare vectors
  - Ensure both value and index vectors get compatible `NestedLayoutAttr`
- [ ] **3.3** Add vectorization tests

### Phase 4: Distribution
- [ ] **4.1** Add `DistributeArgCompare` pattern in `GPUNestedLayoutDistributionPatterns.cpp`
  - Handle local reduction (thread-local elements)
  - Handle thread reduction (across threads in subgroup)
  - Create `amdgpu.subgroup_arg_compare` for subgroup reduction
- [ ] **4.2** Implement `doThreadArgReduction` for value-index pairs
- [ ] **4.3** Add distribution tests

### Phase 5: DPP Lowering
- [ ] **5.1** Add `SubgroupArgCompareToROCDL` pattern in `AMDGPUToROCDL.cpp`
- [ ] **5.2** Implement optimized path for argmax (`isArgMax()`)
  - Use `arith.maximumf` + index tracking
- [ ] **5.3** Implement optimized path for argmin (`isArgMin()`)
  - Use `arith.minimumf` + index tracking
- [ ] **5.4** Implement general path for custom comparators
  - Clone comparator region at each DPP stage
  - Handle tie detection with forward/reverse comparison
- [ ] **5.5** Add chipset-specific handling (gfx9 vs gfx10+)
  - GFX9: `row_bcast_15`, `row_bcast_31`
  - GFX10+: `permlanex16`
- [ ] **5.6** Add ballot-based final winner selection
- [ ] **5.7** Add DPP lowering tests

### Phase 6: Region Inlining Infrastructure
- [ ] **6.1** Implement `inlineComparator` helper function
  - Clone region with value mappings
  - Return comparison result
- [ ] **6.2** Implement `computeTie` helper for tie detection
  - Forward and reverse comparison
  - Return true if values are "equal"
- [ ] **6.3** Implement `analyzeComparator` for pattern detection
  - Detect ArgMax, ArgMin, Custom patterns

### Phase 7: Testing
- [ ] **7.1** Unit tests
  - Config selection tests
  - Vectorization tests
  - Distribution tests
  - DPP lowering tests
- [ ] **7.2** Integration tests
  - End-to-end argmax tests
  - End-to-end argmin tests
  - Custom comparator tests
- [ ] **7.3** Performance benchmarks
  - Compare with ukernel implementation
  - Test various reduction sizes

### Phase 8: Documentation
- [ ] **8.1** Update VectorDistribute pipeline documentation
- [ ] **8.2** Add examples for ArgCompareOp usage
- [ ] **8.3** Document custom comparator limitations

---

## Key Files to Modify

| File | Change |
|------|--------|
| `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp` | Add ArgCompareOp dispatch |
| `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ReductionConfigUtils.cpp` | Add `setArgCompareReductionConfig` |
| `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h` | Add function declaration |
| `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/VectorizeIREELinalgExtOps.cpp` | Add vectorization pattern |
| `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp` | Add distribution pattern |
| `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPU.td` (or IREE location) | Add `subgroup_arg_compare` op |
| `mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp` | Add DPP lowering pattern |

## Reference Implementation

The ROCM argmax ukernel demonstrates the algorithm:
- `compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c`

Key pattern:
```c
// 1. DPP-based value reduction
for (int i = 1; i < warpSize; i *= 2) {
  wgMax = __builtin_fmaxf(__shfl_xor_f(wgMax, i), wgMax);
}

// 2. Ballot to find winners
uint64_t laneHasMaxValmask = __ballot(wgMax == laneMax);

// 3. Tie-breaking with min index
if (__builtin_popcountll(laneHasMaxValmask) > 1) {
  int64_t indexVal = wgMax == laneMax ? laneResult : INT64_MAX;
  laneResult = __ockl_wfred_min_i64(indexVal);
}
```

## Design Document

See [ArgCompareOp_VectorDistribute_Design.md](ArgCompareOp_VectorDistribute_Design.md) for detailed design.

## Labels

- `enhancement`
- `codegen`
- `gpu`
- `amdgpu`
