# IREE Argmax (bf16 → i32/i64) Ukernel Test

This project demonstrates how to compile and run a `bf16 → i32` `argmax` kernel using IREE's HIP backend with microkernel support on AMD GPUs (e.g., MI210, MI300X).

---

## 📁 Files

| File                      | Description                                        |
|---------------------------|----------------------------------------------------|
| `ukernel_bf16_i64.mlir`   | Input example mlir for bf16 and i64                |
| `ukernel_bf16_i32.mlir`   | Input example mlir for bf16 and i32                |          
| `generate_validate_v1.py` | Python script to generate bash script              |
| `run_argmax_bf16.sh`      | Auto-generated bash script to run the test         |


(Note: Since NPY file format does not support the bf16 data type, we use a Python script to generate bash scripts for running the argmax test, with the bf16 input data hardcoded inline.)

---

## Compilation
Using `ukernel_bf16_i32.mlir` as the example
```sh
iree-compile ukernel_bf16_i32.mlir \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx1100 \
  --iree-hip-enable-ukernels=argmax \
  --verify-ir \
  -o ukernel_argmax_i32.vmfb
```

## Script Generation

```
python generate_validate_v1.py
```
This creates `run_argmax_bf16.sh` with a properly formatted inline input for IREE.

## Running the Module

```
bash run_argmax_bf16.sh
```
Expected behavior: the returned index should point to the location of the largest value (e.g., 53.0 at index 98 if input length permits).
## 🛠️ Requirements

- IREE built with `ROCM` backend and microkernel support

Install PyTorch if needed:

```bash
pip install torch
