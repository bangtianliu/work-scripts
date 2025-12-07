#config = #iree_codegen.lowering_config<tile_sizes = [[1, 128]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [128, 1, 1] subgroup_size = 64>
module {
  func.func @test_simple_argmax_dispatch_0_arg_compare_4x1x1336xf16() attributes {translation_info = #translation} {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFC00 : f16
    %c-1 = arith.constant -1 : index
    %c1336 = arith.constant 1336 : index
    %c768 = arith.constant 768 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x128256xf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<4x1x96xf16, #hal.descriptor_type<storage_buffer>>
    %2 = amdgpu.fat_raw_buffer_cast %1 resetOffset : memref<4x1x96xf16, #hal.descriptor_type<storage_buffer>> to memref<4x1x96xf16, #amdgpu.address_space<fat_raw_buffer>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c768) flags(Indirect) : memref<4x1x96xi32, strided<[96, 96, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
    %4 = amdgpu.fat_raw_buffer_cast %3 resetOffset : memref<4x1x96xi32, strided<[96, 96, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<4x1x96xi32, #amdgpu.address_space<fat_raw_buffer>>
    %5 = tensor.empty() : tensor<4x1x96xi32>
    %6 = tensor.empty() : tensor<4x1x96xf16>
    %7 = tensor.empty() : tensor<4x1xi32>
    %8 = tensor.empty() : tensor<4x1xf16>
    %9 = linalg.fill {lowering_config = #config} ins(%cst : f16) outs(%8 : tensor<4x1xf16>) -> tensor<4x1xf16>
    %10 = linalg.fill {lowering_config = #config} ins(%c0_i32 : i32) outs(%7 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %11:2 = scf.forall (%arg0) = (0) to (128256) step (1336) shared_outs(%arg1 = %6, %arg2 = %5) -> (tensor<4x1x96xf16>, tensor<4x1x96xi32>) {
      %12 = arith.cmpi slt, %arg0, %c0 : index
      %13 = arith.subi %c-1, %arg0 : index
      %14 = arith.select %12, %13, %arg0 : index
      %15 = arith.divsi %14, %c1336 : index
      %16 = arith.subi %c-1, %15 : index
      %17 = arith.select %12, %16, %15 : index
      %18 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, %arg0], sizes = [4, 1, 1336], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x128256xf16>> -> tensor<4x1x1336xf16>
      %extracted_slice = tensor.extract_slice %arg1[0, 0, %17] [4, 1, 1] [1, 1, 1] : tensor<4x1x96xf16> to tensor<4x1x1xf16>
      %broadcasted = linalg.broadcast ins(%9 : tensor<4x1xf16>) outs(%extracted_slice : tensor<4x1x1xf16>) dimensions = [2]  {lowering_config = #config}
      %extracted_slice_0 = tensor.extract_slice %broadcasted[0, 0, 0] [4, 1, 1] [1, 1, 1] : tensor<4x1x1xf16> to tensor<4x1xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[0, 0, %17] [4, 1, 1] [1, 1, 1] : tensor<4x1x96xi32> to tensor<4x1x1xi32>
      %broadcasted_2 = linalg.broadcast ins(%10 : tensor<4x1xi32>) outs(%extracted_slice_1 : tensor<4x1x1xi32>) dimensions = [2]  {lowering_config = #config}
      %extracted_slice_3 = tensor.extract_slice %broadcasted_2[0, 0, 0] [4, 1, 1] [1, 1, 1] : tensor<4x1x1xi32> to tensor<4x1xi32>
      %19 = arith.muli %17, %c1336 : index
      %20:2 = scf.forall (%arg3, %arg4) = (0, 0) to (4, 1) step (1, 128) shared_outs(%arg5 = %extracted_slice_0, %arg6 = %extracted_slice_3) -> (tensor<4x1xf16>, tensor<4x1xi32>) {
        %extracted_slice_4 = tensor.extract_slice %18[%arg3, 0, 0] [1, 1, 1336] [1, 1, 1] : tensor<4x1x1336xf16> to tensor<1x1x1336xf16>
        %extracted_slice_5 = tensor.extract_slice %arg5[%arg3, 0] [1, 1] [1, 1] : tensor<4x1xf16> to tensor<1x1xf16>
        %extracted_slice_6 = tensor.extract_slice %arg6[%arg3, 0] [1, 1] [1, 1] : tensor<4x1xi32> to tensor<1x1xi32>
        %21:2 = iree_linalg_ext.arg_compare {lowering_config = #config} dimension(2) ins(%extracted_slice_4 : tensor<1x1x1336xf16>) outs(%extracted_slice_5, %extracted_slice_6 : tensor<1x1xf16>, tensor<1x1xi32>) index_base(%19 : index) {
        ^bb0(%arg7: f16, %arg8: f16):
          %22 = arith.cmpf ogt, %arg7, %arg8 : f16
          iree_linalg_ext.yield %22 : i1
        } -> tensor<1x1xf16>, tensor<1x1xi32>
        %cast = tensor.cast %21#1 : tensor<1x1xi32> to tensor<1x?xi32>
        %cast_7 = tensor.cast %21#0 : tensor<1x1xf16> to tensor<1x?xf16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %cast_7 into %arg5[%arg3, %c0] [1, %c1] [1, 1] : tensor<1x?xf16> into tensor<4x1xf16>
          tensor.parallel_insert_slice %cast into %arg6[%arg3, %c0] [1, %c1] [1, 1] : tensor<1x?xi32> into tensor<4x1xi32>
        }
      } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20#0 into %arg1[0, 0, %17] [4, 1, 1] [1, 1, 1] : tensor<4x1xf16> into tensor<4x1x96xf16>
        tensor.parallel_insert_slice %20#1 into %arg2[0, 0, %17] [4, 1, 1] [1, 1, 1] : tensor<4x1xi32> into tensor<4x1x96xi32>
      }
    } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
    iree_codegen.store_to_buffer %11#0, %2 : tensor<4x1x96xf16> into memref<4x1x96xf16, #amdgpu.address_space<fat_raw_buffer>>
    iree_codegen.store_to_buffer %11#1, %4 : tensor<4x1x96xi32> into memref<4x1x96xi32, #amdgpu.address_space<fat_raw_buffer>>
    return
  }
}