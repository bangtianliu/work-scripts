module @llm {
  // Based on: https://github.com/nod-ai/shark-ai/blob/bb82cc932f945ba69ec8ebbf7b3fc8f7370e2b5c/sharktank/sharktank/examples/export_paged_llm_v1.py#L135
  util.func private @module.prefill_bs4(
    // tokens: tensor<4xSEQ_LENxi64>
    // SEQ_LEN is a dyn_dim which is a multiple of 32.
    // 4 here is batch size.
    %arg0: !hal.buffer_view, 
    // seq_lens: tensor<4xi64>
    %arg1: !hal.buffer_view, 
    // seq_block_ids: tensor<4x(SEQ_LEN/32)xi64>
    %arg2: !hal.buffer_view, 
    // cache: mutable_buffer<NUM_PAGESx2097152xf16>
    // 2097152 = page size required for llama 8b fp16 attention.
    %arg3: !hal.buffer_view) 
    -> !hal.buffer_view attributes {iree.abi.stub}

  // Based on: https://github.com/nod-ai/shark-ai/blob/bb82cc932f945ba69ec8ebbf7b3fc8f7370e2b5c/sharktank/sharktank/examples/export_paged_llm_v1.py#L181C20-L181C26
  util.func private @module.decode_bs4(
    // tokens: tensor<4x1xi64>
    %arg0: !hal.buffer_view, 
    // seq_lens: tensor<4xi64>
    %arg1: !hal.buffer_view, 
    // start_positions: tensor<4xi64>
    %arg2: !hal.buffer_view, 
    // seq_block_ids: tensor<4x(SEQ_LEN/32)xi64>
    %arg3: !hal.buffer_view, 
    // cache: mutable_buffer<NUM_PAGESx2097152xf16>
    %arg4: !hal.buffer_view) 
    -> !hal.buffer_view attributes {iree.abi.stub}

  util.func private @gather_token_ids(
    %tokens: tensor<4x?x128256xf16>, 
    %indices: tensor<4xi64>) 
    -> tensor<4x1x128256xf16> {
    %output = tensor.empty() : tensor<4x1x128256xf16>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(bs, unit, ids) -> (bs)>,
        affine_map<(bs, unit, ids) -> (bs, unit, ids)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%indices : tensor<4xi64>)
      outs(%output : tensor<4x1x128256xf16>) {
        ^bb0(%in_index: i64, %out_elem: f16):
          %iv0 = linalg.index 0 : index
          %in_index_cast = arith.index_cast %in_index : i64 to index
          %iv2 = linalg.index 2 : index
          %gathered = tensor.extract %tokens[%iv0, %in_index_cast, %iv2] : tensor<4x?x128256xf16>
          linalg.yield %gathered : f16
      } -> tensor<4x1x128256xf16>
    util.return %result : tensor<4x1x128256xf16>
  }

  util.func private @argmax(%input : tensor<4x1x128256xf16>) -> tensor<4x1xi32> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -3.402820e+38 : f16

    %out_value_empty = tensor.empty() : tensor<4x1xf16>
    %out_index_empty = tensor.empty() : tensor<4x1xi32>
    %out_value = linalg.fill ins(%cst : f16) outs(%out_value_empty : tensor<4x1xf16>) -> tensor<4x1xf16>
    %out_index = linalg.fill ins(%c0_i32 : i32) outs(%out_index_empty : tensor<4x1xi32>) -> tensor<4x1xi32>

    %out:2 = iree_linalg_ext.arg_compare
      dimension(2)
      ins(%input : tensor<4x1x128256xf16>)
      outs(%out_value, %out_index : tensor<4x1xf16>, tensor<4x1xi32>) {
      ^bb0(%a: f16, %b: f16):
        %cmp = arith.cmpf ogt, %a, %b : f16
        iree_linalg_ext.yield %cmp : i1
    } -> tensor<4x1xf16>, tensor<4x1xi32>

    util.return %out#1 : tensor<4x1xi32>
  }

  util.func private @is_equal(%a : tensor<4x?xi32>, %b : tensor<4x?xi32>) -> i1 {
    %out = arith.constant dense<1> : tensor<i1>
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> ()>
      ],
      iterator_types = ["reduction", "reduction"]
    } ins(%a, %b : tensor<4x?xi32>, tensor<4x?xi32>)
      outs(%out : tensor<i1>) {
      ^bb0(%in_a: i32, %in_b: i32, %acc: i1):
        %cmp = arith.cmpi eq, %in_a, %in_b : i32
        %and = arith.andi %acc, %cmp : i1
        linalg.yield %and : i1
      } -> tensor<i1>
    %extracted = tensor.extract %output[] : tensor<i1>
    util.return %extracted : i1
  }

  util.func @greedy_decoder(
    %tokens: tensor<4x?xi64>,
    %seq_lens: tensor<4xi64>,
    %page_ids: tensor<4x?xi64>,
    %steps: i64,
    %kv_cache_buf: !hal.buffer_view) -> tensor<4x?xi32> {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i64_splat = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi64>

    %padded_seq_len = tensor.dim %tokens, %c1 : tensor<4x?xi64>
    %num_blocks = tensor.dim %page_ids, %c1 : tensor<4x?xi64>

    // Prefill.
    %tokens_buf = hal.tensor.export %tokens : tensor<4x?xi64>{%padded_seq_len} -> !hal.buffer_view
    %seq_lens_buf = hal.tensor.export %seq_lens : tensor<4xi64> -> !hal.buffer_view
    %seq_block_ids_buf = hal.tensor.export %page_ids : tensor<4x?xi64>{%num_blocks} -> !hal.buffer_view
    %prefill_output_buf = util.call @module.prefill_bs4(
      %tokens_buf,
      %seq_lens_buf,
      %seq_block_ids_buf,
      %kv_cache_buf) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
    %prefill_output = hal.tensor.import %prefill_output_buf : !hal.buffer_view -> tensor<4x?x128256xf16>{%padded_seq_len}

    %last_token_indices = arith.subi %seq_lens, %c1_i64_splat : tensor<4xi64>
    %last_tokens = util.call @gather_token_ids(%prefill_output, %last_token_indices) : (tensor<4x?x128256xf16>, tensor<4xi64>) -> tensor<4x1x128256xf16>
    %init_token = util.call @argmax(%last_tokens) : (tensor<4x1x128256xf16>) -> tensor<4x1xi32>

    %steps_index = arith.index_cast %steps : i64 to index
    %num_output_tokens = arith.addi %steps_index, %c1 : index
    %output_tokens = tensor.empty(%num_output_tokens) : tensor<4x?xi32>

    %init_output_tokens = tensor.insert_slice 
      %init_token into %output_tokens[0, 0][4, 1][1, 1] 
      : tensor<4x1xi32> into tensor<4x?xi32>

    %final_outputs, %final_token, %final_seq_lens, %final_positions = 
    scf.for %i = %c0 to %steps_index step %c1 
      iter_args(%tokens_iter = %init_output_tokens, 
                %last_token = %init_token,
                %cur_seq_lens = %seq_lens,
                %curr_positions = %last_token_indices) 
      -> (tensor<4x?xi32>, tensor<4x1xi32>, tensor<4xi64>, tensor<4xi64>) {

      %start_positions = arith.addi %curr_positions, %c1_i64_splat : tensor<4xi64>
      %last_token_i64 = arith.extsi %last_token : tensor<4x1xi32> to tensor<4x1xi64>
      %new_seq_lens = arith.addi %cur_seq_lens, %c1_i64_splat : tensor<4xi64>

      // Decode.
      %last_token_buf = hal.tensor.export %last_token_i64 : tensor<4x1xi64> -> !hal.buffer_view
      %new_seq_lens_buf = hal.tensor.export %new_seq_lens : tensor<4xi64> -> !hal.buffer_view
      %start_positions_buf = hal.tensor.export %start_positions : tensor<4xi64> -> !hal.buffer_view
      %decode_output_buf = util.call @module.decode_bs4(
        %last_token_buf,
        %new_seq_lens_buf,
        %start_positions_buf,
        %seq_block_ids_buf,
        %kv_cache_buf) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
      %decode_output = hal.tensor.import %decode_output_buf : !hal.buffer_view -> tensor<4x1x128256xf16>

      %next_token = util.call @argmax(%decode_output) : (tensor<4x1x128256xf16>) -> tensor<4x1xi32>
      %iplus1 = arith.addi %i, %c1 : index
      %new_tokens = tensor.insert_slice 
        %next_token into %tokens_iter[0, %iplus1][4, 1][1, 1] 
        : tensor<4x1xi32> into tensor<4x?xi32>

      scf.yield %new_tokens, %next_token, %new_seq_lens, %start_positions : tensor<4x?xi32>, tensor<4x1xi32>, tensor<4xi64>, tensor<4xi64>
    }

    util.return %final_outputs : tensor<4x?xi32>
  }

  util.func public @test_greedy_decoder() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.0 : f16
    %true = arith.constant 1 : i1

    // Pre-padded tokens.
    %tokens = arith.constant dense<[
      // """<|begin_of_text|>Name the capital of the United States.<|eot_id|>"""
      [128000, 128000, 678, 279, 6864, 315, 279, 3723, 4273, 13, 128009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      // """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
      // Hey!! Expect the response to be printed as comma separated values.<|eot_id|>
      // 
      // <|start_header_id|>user<|end_header_id|>
      // Give me the first 10 prime numbers<|eot_id|>
      // 
      // <|start_header_id|>assistant<|end_header_id|>""",
      [128000, 128000, 128006, 9125, 128007, 198, 19182, 3001, 33185, 279, 2077, 311, 387, 17124, 439, 32783, 19180, 2819, 13, 128009, 271, 128006, 882, 128007, 198, 36227, 757, 279, 1176, 220, 605, 10461, 5219, 128009, 271, 128006, 78191, 128007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      // """IREE is an MLIR-based end-to-end compiler and runtime that lowers Machine
      // Learning (ML) models to a unified IR that scales up to meet the needs of the
      // datacenter and down to satisfy the constraints and special considerations of
      // mobile and edge deployments. IREE supports importing from a variety of ML
      // frameworks: JAX ONNX PyTorch TensorFlow and TensorFlow Lite""",
      [128000, 40, 6731, 374, 459, 20187, 2871, 6108, 842, 4791, 13368, 19979, 323, 15964, 430, 73115, 13257, 198, 48567, 320, 2735, 8, 4211, 311, 264, 43790, 16646, 430, 29505, 709, 311, 3449, 279, 3966, 315, 279, 198, 695, 3133, 323, 1523, 311, 27651, 279, 17413, 323, 3361, 38864, 315, 198, 15280, 323, 6964, 72642, 13, 358, 6731, 11815, 50995, 505, 264, 8205, 315, 20187, 198, 3879, 82, 25, 622, 3027, 6328, 44404, 5468, 51, 22312, 96086, 323, 96086, 41965, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      // "What is the answer to life?",
        [128000, 3923, 374, 279, 4320, 311, 2324, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      ]> : tensor<4x96xi64>

    // Seq lengths of unpadded input.
    %seq_lens = arith.constant dense<[11, 38, 79, 8]> : tensor<4xi64>

    // Allocate 3 blocks for each sequence in the batch.
    // Skip page_id = 0.
    %page_ids = arith.constant dense<[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
      ]> : tensor<4x3xi64>

    // Allocate a KV Cache with 16 pages.
    %kv_cache = tensor.empty() : tensor<32x2097152xf16>
    %kv_cache_buf = hal.tensor.export %kv_cache : tensor<32x2097152xf16> -> !hal.buffer_view

    // Number of decoding steps.
    %steps = arith.constant 14 : i64

    %tokens_dyn = tensor.cast %tokens : tensor<4x96xi64> to tensor<4x?xi64>
    %page_ids_dyn = tensor.cast %page_ids : tensor<4x3xi64> to tensor<4x?xi64>
    %output = util.call @greedy_decoder(
      %tokens_dyn,
      %seq_lens,
      %page_ids_dyn,
      %steps,
      %kv_cache_buf) : (tensor<4x?xi64>, tensor<4xi64>, tensor<4x?xi64>, i64, !hal.buffer_view) -> tensor<4x?xi32>

    // Outputs checked using: https://gist.github.com/Groverkss/b311f87d45d5e0098ec7ebf39b0e5b87
    // You can convert the output back to tokens using the tokenizer in the gist.
    %expected_output = arith.constant dense<[
      // <|start_header_id|>assistant<|end_header_id|>\n\nThe capital of the United States is Washington, D.C
      [128006, 78191, 128007, 271, 791, 6864, 315, 279, 3723, 4273, 374, 6652, 11, 423, 732],
      // \n\n2, 3, 5, 7, 11,
      [271, 17, 11, 220, 18, 11, 220, 20, 11, 220, 22, 11, 220, 806, 11],
      // .\n\nIREE is designed to be a drop-in replacement for existing ML frameworks
      [382, 40, 6731, 374, 6319, 311, 387, 264, 6068, 3502, 14039, 369, 6484, 20187, 49125],
      // The universe? Everything?\nThe answer is 42, according to Douglas Adams
      [578, 15861, 30, 20696, 5380, 791, 4320, 374, 220, 2983, 11, 4184, 311, 31164, 27329]
    ]> : tensor<4x15xi32>

    %expected_output_dyn = tensor.cast %expected_output : tensor<4x15xi32> to tensor<4x?xi32>

    %is_equal = util.call @is_equal(%output, %expected_output_dyn) : (tensor<4x?xi32>, tensor<4x?xi32>) -> i1
    %failed = arith.xori %is_equal, %true : i1
    scf.if %failed {
      %num_tokens = tensor.dim %output, %c1 : tensor<4x?xi32>
      flow.tensor.trace "Expected Output" = [
        %expected_output_dyn : tensor<4x?xi32>{%num_tokens}
      ]
      flow.tensor.trace "Actual Output" = [
        %output : tensor<4x?xi32>{%num_tokens}
      ]
      %status = arith.constant 1 : i32
      util.status.check_ok %status, "Output Mismatch"
    }

    util.return
  }
}
