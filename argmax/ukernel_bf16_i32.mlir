func.func @forward_bf16(%arg0: tensor<1x?xbf16>) -> tensor<1x1xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0xFF80 : bf16  // -inf for bf16
  %c-1_i32 = arith.constant -1 : i32

  %init_idx = tensor.empty() : tensor<1xi32>
  %init_val = tensor.empty() : tensor<1xbf16>
  %filled_idx = linalg.fill ins(%c0_i32 : i32) outs(%init_idx : tensor<1xi32>) -> tensor<1xi32>
  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<1xbf16>) -> tensor<1xbf16>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<1x?xbf16>) outs(%filled_val, %filled_idx : tensor<1xbf16>, tensor<1xi32>) {
    ^bb0(%in: bf16, %out: bf16, %out_2: i32):
      %idx = linalg.index 1 : index
      %idx_i32 = arith.index_cast %idx : index to i32
      %max = arith.maximumf %in, %out : bf16
      %pred = arith.cmpf ogt, %in, %out : bf16
      %selected = arith.select %pred, %idx_i32, %out_2 : i32
      linalg.yield %max, %selected : bf16, i32
  } -> (tensor<1xbf16>, tensor<1xi32>)

  %expanded_1 = tensor.expand_shape %result#1 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
  return %expanded_1 : tensor<1x1xi32>
}