# A5 ND-to-NZ destination storage bounds

Date: 2026-10-15

Status: proposed

## Context

The A5 EmitC implementation of `TMOV` from ND vector storage to NZ vector
storage uses `align16(dst valid rows)` as its C0-block stride. Its row loop
uses the source valid rows. A declared `8x128xf16` destination occupies 2048
bytes, but this conversion writes C0 blocks across a 3840-byte span. Adjacent
live allocations can be overwritten even without address reuse.

An independent FA reproduction matched every zero-input output bit to this
overwrite. Upstream `master@29d24227d` still accepts that input. A same-input
control with Cube M32 and two AIV tiles of 16 rows passed the independent
golden on A5/CANN 9.2.0-beta.1. Merely padding the NZ destination while keeping
the original pipe geometry did not pass.

## Decision

Add an A5 ND-to-NZ storage bound check to the existing TMOV verifier. Check
before EmitC lowering and before allocations acquire addresses. Do not change
IR syntax, annotations, scheduling, or memory planning.

For positive column count, bound the write span in bytes by:

```text
C0 = 32 / storage_element_bytes
blocks = ceil(dst_valid_cols / C0)
stride_rows = align16(dst_valid_rows) + (compact == RowPlusOne ? 1 : 0)
span = ((blocks - 1) * stride_rows + src_valid_rows) * 32
```

The final partial C0 block is counted in full. Dynamic valid dimensions use
their declared physical dimensions as upper bounds; this does not prove
runtime positivity or all dynamic access conditions. Reject statically zero
valid rows with nonzero columns because the implementation's row loop
subtracts one. Zero columns produce no writes.

Compare against checked physical capacity. Account for the existing col-major
RowPlusOne allocation formula. Reject arithmetic overflow with an ordinary
diagnostic. Packed FP4 and unrecognized storage widths retain their existing
verification behavior; this check does not add backend support for them.

The diagnostic includes required and available bytes. It also tells the caller
to keep pipe split geometry consistent with physical rows. The check proves
only this destination storage bound. It does not certify read strides, alias
safety, pipe pairing, synchronization, or numerical correctness.

## Alternatives and compatibility

- Increasing only PlanMemory's allocation size does not correct the source NZ
  read stride or a peer's physical split geometry.
- Automatically changing tile rows would require rewriting dependent views,
  pipe producer/consumer geometry, slot sizes, and possibly iteration domains.
  That is a separate transformation with its own correctness requirements.
- A blanket requirement for 16 physical rows would unnecessarily reject a
  small destination with only one C0 column block. Check the actual span.

Previously accepted conversions whose writes exceed declared storage now fail
at verification. Existing A3 and other TMOV layout forms are unchanged. Safe
16-row, padded-valid-row, one-C0-block, and RowPlusOne forms remain accepted.

## Validation

Add focused lit checks for FP16/FP32 byte spans, RowPlusOne, dynamic upper
bounds, overflow, empty row handling, and safe boundary cases. Retain TMOV,
normalization, memory-plan and pipe regressions. Keep device evidence for the
original FA and the M32/AIV16 control separate. A diagnostic improvement is
not a claim that the original FA has been automatically repaired.
