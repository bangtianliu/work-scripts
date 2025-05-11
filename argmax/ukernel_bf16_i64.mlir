func.func @forward_bf16(%arg0: tensor<1x?xbf16>) -> tensor<1x1xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF80 : bf16  // -inf for bf16
  %c-1_i64 = arith.constant -1 : i64

  %init_idx = tensor.empty() : tensor<1xi64>
  %init_val = tensor.empty() : tensor<1xbf16>
  %filled_idx = linalg.fill ins(%c0_i64 : i64) outs(%init_idx : tensor<1xi64>) -> tensor<1xi64>
  %filled_val = linalg.fill ins(%cst : bf16) outs(%init_val : tensor<1xbf16>) -> tensor<1xbf16>

  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<1x?xbf16>) outs(%filled_val, %filled_idx : tensor<1xbf16>, tensor<1xi64>) {
    ^bb0(%in: bf16, %out: bf16, %out_2: i64):
      %idx = linalg.index 1 : index
      %idx_i64 = arith.index_cast %idx : index to i64
      %max = arith.maximumf %in, %out : bf16
      %pred = arith.cmpf ogt, %in, %out : bf16
      %selected = arith.select %pred, %idx_i64, %out_2 : i64
      linalg.yield %max, %selected : bf16, i64
  } -> (tensor<1xbf16>, tensor<1xi64>)
  %expanded_1 = tensor.expand_shape %result#1 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
  return %expanded_1 : tensor<1x1xi64>
}