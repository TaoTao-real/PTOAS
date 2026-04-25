# CA Model ISA Verification Skill

> How to create ST test cases, run them on the CA model (cycle-accurate simulator), and analyze instruction logs for latency and correctness verification.

---

## Overview

This skill covers the complete workflow for verifying PTO-ISA instructions using the CANN simulator (CA model):

1. **Create/Select an ST test case**
2. **Build and run on NPU SIM**
3. **Locate and interpret instruction logs**
4. **Extract latency data**
5. **Verify hex correctness via UB dumps**

---

## Prerequisites

### Environment Setup

```bash
cd ~/pto-isa
source /usr/local/Ascend/cann/set_env.sh
```

### Build Fix (CANN 9.0.0)

If using CANN 9.0.0-alpha.1, ensure the namespace fix is applied:

```bash
# Check if fix is applied
grep '__cce_simd::RoundRType' ~/pto-isa/include/pto/npu/a5/TCvt.hpp

# If not, apply:
# Change "using ::RoundRType;" to "using __cce_simd::RoundRType;" etc.
```

---

## Step 1: Create/Select a Test Case

### Existing Test Structure

Tests are located in `~/pto-isa/tests/st/` with naming convention `t<operation>`:

```
tests/st/
├── tadd/           # Vector add
├── tmul/           # Vector multiply
├── tload/          # Load operations
├── tgather/        # Gather operations
├── tcvt/           # Type conversions
└── ...
```

### Test File Structure

Each test directory contains:

```
tests/st/tadd/
├── CMakeLists.txt
├── test_tadd.cpp   # Google Test cases
└── inc/
    └── test_tadd.hpp  # Test parameters & configs
```

### Creating a New Test Case

1. **Copy an existing similar test:**
   ```bash
   cp -r tests/st/tadd tests/st/tnewop
   ```

2. **Modify the test source** (`test_tnewop.cpp`):
   ```cpp
   #include "gtest/gtest.h"
   #include "pto/npu/a5/TNewOp.hpp"  // Your new operation header
   
   class TNEWOPTest : public ::testing::Test {
   protected:
       void SetUp() override {
           // Initialize input buffers, parameters
       }
       void TearDown() override {
           // Cleanup
       }
   };
   
   TEST_F(TNEWOPTest, case_float_64x64) {
       // Setup input data
       // Call the operation
       // Verify output matches expected
   }
   ```

3. **Update CMakeLists.txt:**
   ```cmake
   set(TEST_NAME tnewop)
   # Add sources, includes, link libraries
   ```

---

## Step 2: Run Test on CA Model

### Command Syntax

```bash
# Run full test suite:
python3 tests/script/run_st.py -r sim -v a5 -t <testcase>

# Run specific test case:
python3 tests/script/run_st.py -r sim -v a5 -t <testcase> -g "<TestSuite>.<case_name>"
```

### Examples

```bash
# Full tadd suite
python3 tests/script/run_st.py -r sim -v a5 -t tadd

# Specific tadd case
python3 tests/script/run_st.py -r sim -v a5 -t tadd -g "TADDTest.case_float_64x64_64x64_64x64_64x64"

# Multiple tests with output capture
python3 tests/script/run_st.py -r sim -v a5 -t tmul 2>&1 | tee logs/tmul.log
```

### SOC Versions

| SOC Version | Description |
|-------------|-------------|
| `a5` | Ascend 910 (A5 architecture) |
| `Ascend910_9599` | Full chip model |

### Run Mode Options

| Mode | Description |
|------|-------------|
| `sim` | Run on cycle-accurate simulator (CA model) |
| `npu` | Run on real NPU hardware |
| `cpu` | Run on CPU simulator |

---

## Step 3: Locate Output Logs

### Output Directory Structure

After running a test, logs are generated in:

```
build/tests/st/<testcase>/
└── core0.veccore0.instr_log.dump        # Main instruction trace
└── core0.veccore0.instr_popped_log.dump # Instruction pipeline log
└── core0.veccore0.ub.rd_log.dump        # UB read log (hex dumps)
└── core0.veccore0.ub.wr_log.dump        # UB write log (hex dumps)
```

### Copying Logs for Analysis

```bash
mkdir -p ~/npu_skills/pto-isa/verification/logs/<test_name>
cp build/tests/st/<testcase>/core0.veccore0.*.dump \
   ~/npu_skills/pto-isa/verification/logs/<test_name>/
```

---

## Step 4: Parse Instruction Logs for Latency

### Log Format

The `instr_log.dump` file contains cycle-by-cycle instruction execution:

```
[info] [CYCLE] (PC: 0x...) PIPE : (Binary: 0x...) (ID: NNNNNN) INSTR_NAME params
```

### Key Fields

| Field | Description |
|-------|-------------|
| `[CYCLE]` | Clock cycle when instruction issued/completed (10-digit) |
| `PC` | Program counter address |
| `Binary` | Instruction hex encoding |
| `ID` | Instruction sequence number |
| `PIPE` | Execution pipe: SCALAR, MTE2, RVECLD, RVECEX, RVECST |
| `INSTR_NAME` | Instruction mnemonic |

### Example Log Lines

```
[info] [00002664] (PC: 0x13b3b714) RVECLD   : (Binary: 0x00180008) (ID: 000112) RV_VLD  Vd[0], Sn[6]=0x0, ...
[info] [00002669] (PC: 0x13b3b720) RVECEX   : (Binary: 0x80082780) (ID: 000115) RV_VADD Dtype: F32  Vd[0], Vn[0], Vm[1], ...
[info] [00002676] (PC: 0x13b3b724) RVECST   : (Binary: 0x40280000) (ID: 000116) RV_VST  Vn[0], ...
```

### Calculating Latency

**Latency = (Store complete cycle) - (Load issue cycle)**

For the example above:
- VLD issued at cycle 2664
- VADD issued at cycle 2669 (depends on VLD)
- VST issued at cycle 2676 (depends on VADD)

**VLD latency** = 2669 - 2664 = 5 cycles (+ pipeline dependency)
**VADD latency** = 2676 - 2669 = 7 cycles

### Instruction Mapping Table

| PTO Op | A5 Vector Instruction | Pipe | Typical Latency |
|--------|----------------------|------|-----------------|
| TLOAD | MOV_SRC_TO_DST_ALIGNv2 (MTE2 DMA) | MTE2 | Variable |
| VLD | RV_VLD (NORM) | RVECLD | 9 cycles |
| VST | RV_VST (NORM_B32) | RVECST | 9 cycles |
| VADD | RV_VADD | RVECEX | 7 cycles |
| VMUL | RV_VMUL | RVECEX | 7 cycles |
| VGATHER2 | RV_VGATHER2 | RVECLD | 28 cycles |
| VLD UNPK_B16 | RV_VLD (dist=UNPK_B16) | RVECLD | 9 cycles |
| VLD PK_B32 | RV_VLD (dist=PK_B32) | RVECLD | 9 cycles |

### Parsing Script

```bash
#!/bin/bash
# Extract latency pairs from instruction log

LOG=$1
echo "=== Latency Analysis: $LOG ==="

# Find VLD->VADD pairs
grep -E "RV_VLD|RV_VADD|RV_VST" $LOG | head -20

# Parse cycle numbers
grep "RV_VLD" $LOG | head -5 | while read line; do
  CYCLE=$(echo $line | sed 's/.*\[\([0-9]*\)\].*/\1/')
  ID=$(echo $line | sed 's/.*ID: \([0-9]*\).*/\1/')
  echo "VLD ID:$ID at cycle:$CYCLE"
done
```

---

## Step 5: Verify Hex Correctness via UB Dumps

### UB Write Log Format

The `ub.wr_log.dump` shows memory writes:

```
[info] [CYCLE] INSTR_NAME, pc: 0x..., id: NNN
[info] Address XXXXXXXX = [HHHHHHHH]  Address ...
```

### Example UB Dump

```
[info] [0000002027] MOV_SRC_TO_DST_ALIGNv2, pc: 0x13b3a084, id: 33
[info] Address 00000000 = [40c00000]  Address 00000004 = [40400000]  ...
       Address 00000010 = [40a00000]  Address 00000014 = [40400000]  ...
```

### Interpreting Hex Values

| Hex | Float32 | Description |
|-----|---------|-------------|
| `3f800000` | 1.0 | |
| `40000000` | 2.0 | |
| `40400000` | 3.0 | |
| `40800000` | 4.0 | |
| `40a00000` | 5.0 | |
| `40c00000` | 6.0 | |
| `40e00000` | 7.0 | |
| `41000000` | 8.0 | |
| `41100000` | 9.0 | |

### Verification Steps

1. **Compare UB write addresses** with expected output buffer location
2. **Decode hex to float** and compare with expected values
3. **Check for padding patterns** (e.g., `0x00000000` for zero-pad)

### Python Verification Script

```python
import struct

def hex_to_float(hex_str):
    """Convert hex string to float32"""
    return struct.unpack('!f', bytes.fromhex(hex_str))[0]

def verify_ub_dump(log_path, expected_values, start_addr=0):
    """Verify UB dump matches expected float values"""
    with open(log_path) as f:
        for line in f:
            if 'Address' in line:
                # Parse: Address XXXXXXXX = [HHHHHHHH]
                parts = line.split('Address')
                for part in parts[1:]:
                    if '=' in part:
                        addr_hex, val_hex = part.split('=')
                        addr = int(addr_hex.strip(), 16)
                        val = hex_to_float(val_hex.strip().strip('[]'))
                        
                        idx = (addr - start_addr) // 4
                        if idx < len(expected_values):
                            if abs(val - expected_values[idx]) > 1e-5:
                                print(f"MISMATCH at addr {addr:08x}: got {val}, expected {expected_values[idx]}")
                            else:
                                print(f"OK addr {addr:08x}: {val}")

# Example usage
expected = [6.0, 3.0, 9.0, 4.0, 5.0, 3.0, 9.0, 7.0]  # First 8 values
verify_ub_dump('core0.veccore0.ub.wr_log.dump', expected)
```

---

## Step 6: Correlate Instruction ID to Name

### Finding Instruction by ID

Each instruction has a unique ID assigned during compilation. To find an instruction:

```bash
# Search for specific instruction ID
grep "ID: 000115" core0.veccore0.instr_log.dump
```

Output:
```
[info] [00002669] (PC: 0x13b3b720) RVECEX   : (Binary: 0x80082780) (ID: 000115) RV_VADD Dtype: F32  Vd[0], Vn[0], Vm[1], Pg[1],
```

### Decoding Binary to Instruction

The `Binary: 0xXXXXXXXX` field is the raw instruction encoding. Use the A5 ISA reference to decode:

| Bits | Field |
|------|-------|
| 31:26 | Opcode |
| 25:21 | Dest register |
| 20:16 | Src1 register |
| 15:11 | Src2 register |
| 10:0 | Modifiers/Immediate |

---

## Quick Reference

### Common Test Commands

```bash
# Build and run single test
python3 tests/script/run_st.py -r sim -v a5 -t tadd

# Run with debug output
python3 tests/script/run_st.py -r sim -v a5 -t tadd --debug

# List available tests
ls tests/st/

# Check test result
grep -E "PASS|FAIL" build/tests/st/tadd/*.log
```

### Verified Latencies (A5)

| Instruction | Latency (cycles) | Notes |
|-------------|-----------------|-------|
| VLD NORM | 9 | Normal vector load |
| VLD UNPK_B16 | 9 | Unpack 16-bit |
| VLD PK_B32 | 9 | Pack 32-bit |
| VST NORM_B32 | 9 | Normal vector store |
| VADD F32 | 7 | FP32 add |
| VMUL F32 | 7 | FP32 multiply |
| VGATHER2 B32 | 28 | Indexed gather |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "NOT SUPPORT TASK TYPE 3" warning | Ignore - normal simulator message |
| Test hangs | Check timeout, may need longer for large tiles |
| "namespace" errors | Apply TCvt.hpp fix (see Prerequisites) |
| No output logs | Check build directory, ensure `-r sim` mode |

---

## References

- PTO-ISA repo: `~/pto-isa/`
- A5 ISA reference: `~/pto-isa/docs/`
- Test scripts: `~/pto-isa/tests/script/`
- Verification reports: `~/npu_skills/pto-isa/verification/reports/`

---

*Created: 2026-03-25*
*Author: AI Assistant*
*Version: 1.0*
