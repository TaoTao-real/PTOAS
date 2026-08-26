# Partial RoPE A5 hardware harness

This harness compares the generic PTOAS D objects with the attached explicit
AscendC `__simd_vf__` implementation under one host launcher and one GM
contract.  The caller supplies `vf_rope_attached.h` through `MANUAL_VF_DIR`;
the header's parent directory must contain the matching `compressor_comm.h`.

The fixed contract is FP32 input `[8,512]`, FP32 sin/cos `[8,64]`, BF16 RINT
output `[8,512]`, `baseAddr=448`, for both HALF and INTERLEAVE.
