import numpy as np

# Simulate bf16 buffer (bit patterns in uint16)
batch = 1
reduction_size = 32000
label = 98

data_f32 = np.ones([batch, reduction_size], dtype=np.float32) * 7.0
data_f32[0, label] = 53.0

# Convert to bf16 (keep upper 16 bits)
data_u32 = data_f32.view(np.uint32)
data_bf16 = (data_u32 >> 16).astype(np.uint16)

# Save as raw binary (little-endian)
data_bf16.tofile("input0_bf16.bin")