# ArgCompareOp VectorDistribute Pipeline Support

Support `iree_linalg_ext.arg_compare` through VectorDistribute pipeline on AMD GPU using DPP + ballot.

## Pipeline Flow

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

## TODO

### Phase 1: Infrastructure

- [ ] **AMDGPU Dialect Operation**
  Define `amdgpu.subgroup_arg_compare` operation with comparator region, value/index inputs, and dual outputs. Add helper methods `isArgMax()`, `isArgMin()`, `isCustomComparator()` for pattern detection. Include verifier, printer, and parser.
  - File: Create in IREE's AMDGPU extensions or propose upstream

- [ ] **DPP Lowering**
  Add `SubgroupArgCompareToROCDL` pattern. Implement optimized paths for argmax/argmin using `arith.maximumf`/`minimumf`. Implement general path with region inlining for custom comparators (clone at each DPP stage). Handle chipset differences (gfx9: row_bcast, gfx10+: permlanex16). Add ballot-based winner selection with tie-breaking (smallest index wins).
  - File: New pattern in AMDGPU → ROCDL lowering

### Phase 2: KernelConfig Integration

- [ ] **KernelConfig Integration**
  Add `setArgCompareReductionConfig` in `ReductionConfigUtils.cpp` following the same pattern as `setReductionConfig`. Add declaration in `ConfigUtils.h`. Add ArgCompareOp case in `KernelConfig.cpp:setRootConfig` after LinalgOp handling (around line 2291), gated with `clGPUEnableReductionVectorDistribution` flag.
  - Files: `KernelConfig.cpp`, `ReductionConfigUtils.cpp`, `ConfigUtils.h`

### Phase 3: Vectorization

- [ ] **Vectorization**
  Add ArgCompareOp handling to `GenericVectorizationPass`. Add `IREE::LinalgExt::ArgCompareOp` to candidates list (similar to GatherOp handling). Create custom vectorization function `vectorizeArgCompareOp()`:
  - Read input as vector via `vector.transfer_read`
  - Initialize index vector with `vector.step` for element indices [0, 1, 2, ..., N-1]
  - Generate vectorized reduction with paired (value, index) outputs
  - Alternative: Add pattern to `VectorizeIREELinalgExtOps.cpp` and call from GenericVectorization
  - File: `GenericVectorization.cpp` or `VectorizeIREELinalgExtOps.cpp`

### Phase 4: Layout Configuration and Analysis

- [ ] **Layout Configuration (ToLayoutOp insertion)**
  Add `setArgCompareAnchor()` function similar to `setContractionAnchor()`. Insert `iree_vector_ext.to_layout` ops on ArgCompareOp's vectorized outputs. Configure `NestedLayoutAttr` with appropriate `thread_tile` for reduction dimension. Ensure both value and index vectors get **identical** layouts (they must be distributed the same way).
  - File: `LLVMGPUConfigureTensorLayouts.cpp`

- [ ] **Layout Analysis (propagation)**
  Add forward propagation: when ArgCompareOp source has a layout, project it for the result (similar to `MultiDimReductionOp` handling at lines 108-126). Add backward propagation: propagate layout from results back to accumulator operands (similar to lines 218-221). Key difference from `MultiDimReductionOp`: ArgCompareOp has **two results** (value + index), both need the same projected layout.
  - File: `VectorLayoutAnalysis.cpp`

### Phase 5: Distribution

- [ ] **Distribution**
  Add `DistributeArgCompare` pattern in `GPUNestedLayoutDistributionPatterns.cpp`. Follow pattern of `DistributeMultiReduction` (lines 966-1395). Handle local reduction for thread-local elements (batch × outer × element tiles), then create `amdgpu.subgroup_arg_compare` for subgroup-level reduction with value-index pairs.
  - File: `GPUNestedLayoutDistributionPatterns.cpp`

### Phase 6: Testing

- [ ] **Unit Tests**
  - DPP lowering tests
  - Vectorization tests
  - Distribution tests
  - Layout analysis tests

- [ ] **End-to-End Tests**
  Extend `tests/e2e/linalg_ext_ops/arg_compare.mlir`. Add GPU-specific tests for argmax, argmin, and custom comparator cases with various reduction sizes.

## Key Files

| Component | File |
|-----------|------|
| **Op Definition** | `LinalgExtOps.td:645-770` |
| **GenericVectorizationPass** | `GenericVectorization.cpp` |
| **LinalgExt Vectorization** | `VectorizeIREELinalgExtOps.cpp` |
| **Layout Configuration** | `LLVMGPUConfigureTensorLayouts.cpp` |
| **Layout Analysis** | `VectorLayoutAnalysis.cpp` |
| **Distribution Patterns** | `GPUNestedLayoutDistributionPatterns.cpp` |
| **KernelConfig** | `KernelConfig.cpp` |
| **ReductionConfigUtils** | `ReductionConfigUtils.cpp` |
| **UKernel Reference** | `iree_uk_amdgpu_argmax_f32i64.c` |
| **DPP Lowering Reference** | `SubgroupReduceLowering.cpp` |
| **E2E Tests** | `tests/e2e/linalg_ext_ops/arg_compare.mlir` |

## Reference

- Design: `ArgCompareOp_VectorDistribute_Design.md`
- UKernel: `compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_argmax_f32i64.c`
- Reduction Summary: `reduction_vd/reduction_vector_distribute_summary.md`
