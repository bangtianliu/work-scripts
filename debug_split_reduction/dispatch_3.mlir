module {
  func.func @test_simple_argmax(%input: tensor<4x1x128256xf16>) -> (tensor<4x1xf16>, tensor<4x1xi32>) {
    %cst = arith.constant 0xFC00 : f16  // -inf for f16
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    
    // Initialize output tensors
    %init_max_value = tensor.empty() : tensor<4x1xf16>
    %init_max_index = tensor.empty() : tensor<4x1xi32>
    
    // Fill with initial values
    %max_value_init = linalg.fill ins(%cst : f16) outs(%init_max_value : tensor<4x1xf16>) -> tensor<4x1xf16>
    %max_index_init = linalg.fill ins(%c0_i32 : i32) outs(%init_max_index : tensor<4x1xi32>) -> tensor<4x1xi32>
    
    // Perform argmax on full tensor (dimension 2 = vocab dimension)
    %max_val, %max_idx = iree_linalg_ext.arg_compare dimension(2) 
      ins(%input : tensor<4x1x128256xf16>) 
      outs(%max_value_init, %max_index_init : tensor<4x1xf16>, tensor<4x1xi32>) {
    ^bb0(%lhs: f16, %rhs: f16):
      %cmp = arith.cmpf ogt, %lhs, %rhs : f16
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4x1xf16>, tensor<4x1xi32>
    
    return %max_val, %max_idx : tensor<4x1xf16>, tensor<4x1xi32>
  }
}