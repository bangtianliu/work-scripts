import sys
import torch
import numpy as np

# =============================
# Parse command-line argument
# =============================

if len(sys.argv) != 2:
    print("Usage: python generate_bf16_test.py <length>")
    sys.exit(1)

length = int(sys.argv[1])
if length <= 0:
    raise ValueError("Length must be a positive integer.")

# =============================
# Build input tensor
# =============================

batch = 1
values = [1.0] * length
if length > 98:
    values[98] = 53.0  # inject standout for argmax

tensor_bf16 = torch.tensor(values, dtype=torch.bfloat16).reshape(batch, length)
tensor_f32 = tensor_bf16.to(torch.float32).numpy()

# =============================
# Format for IREE CLI
# =============================

flat_values_str = ",".join(f"{x:.6g}" for x in tensor_f32.flatten())

inline_bash_cmd = f"""#!/bin/bash

iree-run-module \\
  --module=ukernel_argmax.vmfb \\
  --device=hip \\
  --function=forward_bf16 \\
  --input={batch}x{length}xbf16=[[{flat_values_str}]] \\
  --device_allocator=caching
"""

# =============================
# Write shell script
# =============================

output_file = "run_argmax_bf16.sh"
with open(output_file, "w") as f:
    f.write(inline_bash_cmd)

print(f"[✓] Generated script: {output_file}")
