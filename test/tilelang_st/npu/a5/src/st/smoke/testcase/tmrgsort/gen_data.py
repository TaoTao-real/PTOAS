#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
import os
import sys
import struct
import ctypes

# Add parent directory to path for st_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_common import setup_case_rng, save_case_data

from cases import CASES

BLOCK_NUM = 4
STRUCT_SIZE = 8  # bytes per structure (value + index)


def find_and_zero(arr, tar):
    for item in arr:
        if not isinstance(item, (np.floating)):
            return -1
    if not all(isinstance(x, (np.floating)) for x in arr):
        raise ValueError("The input must be a list of numbers.")
    if not isinstance(tar, (np.floating)):
        return -1

    n = len(arr)
    for i in range(n - 1, -1, -1):
        if arr[i] == tar:
            for j in range(i + 1, n):
                arr[j] = 0
            return i
    return -1


def zero_after_index(arr, i):
    if i < 0 or i >= len(arr):
        return
    for j in range(i + 1, len(arr)):
        arr[j] = 0


def handle_exhausted_list(input_num, topk_sorted_output_global, topk_sorted_idx_global, last_data):
    for i in range(input_num):
        zero_index = find_and_zero(topk_sorted_output_global, last_data[i])
        zero_after_index(topk_sorted_idx_global, zero_index)


def _to_tuple(shape):
    """Convert shape to tuple if needed."""
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


def get_elem_divisor(dtype):
    """Get element divisor based on dtype.

    A structure is 8 bytes:
    - f32 (4 bytes): 8 / 4 = 2 elems per struct
    - f16 (2 bytes): 8 / 2 = 4 elems per struct
    """
    if dtype == np.float16:
        return 4
    return 2


def write_value_index_pair(f, value, index, dtype):
    """Write a (value, index) pair to file.

    Format: value followed by index (uint32).
    For f16: value (2 bytes) + padding (2 bytes) + index (4 bytes).
    For f32: value (4 bytes) + index (4 bytes).
    """
    if dtype == np.float32:
        packed_data = struct.pack('fI', float(value), ctypes.c_uint32(index).value)
        f.write(packed_data)
    elif dtype == np.float16:
        # f16: directly pack value (np.float16), not float(value)
        # Following pto-isa: struct.pack('e2xI', value, ...)
        packed_data = struct.pack('e2xI', value, ctypes.c_uint32(index).value)
        f.write(packed_data)


def gen_golden_single(case):
    """Generate golden data for Format1 (single list internal block sorting).

    Following pto-isa gen_data.py logic exactly:
    - cols = src_cols // elem_divisor (STRUCTURE count, using full src_cols not valid_cols)
    - list_col = block_len // elem_divisor (STRUCTURES per block)
    - block_lens = list_col * 4 (STRUCTURES per vmrgsort4 call)
    - block_lens_floats = block_len * 4 (FLOATS per vmrgsort4 call)

    Process:
    1. Generate random data (cols structures)
    2. Reshape into blocks (each list_col structures)
    3. Sort each block internally -> input0.bin
    4. Reshape into groups (each block_lens structures)
    5. Globally sort each group -> golden.bin

    For cases where src_cols > valid_cols (padding), generate full src_cols with zeros padding.
    """
    dtype = case["dtype"]
    src_shape = _to_tuple(case["src_shape"])
    dst_shape = _to_tuple(case["dst_shape"])
    valid_shape = _to_tuple(case["valid_shape"])
    block_len = case["block_len"]

    src_rows, src_cols = src_shape
    valid_rows, valid_cols = valid_shape

    # Get element divisor based on dtype (2 for f32, 4 for f16)
    elem_divisor = get_elem_divisor(dtype)

    # Use FULL src_cols for file size (matching pto-isa kGCols)
    cols = src_cols // elem_divisor  # total structures in file
    valid_structs = valid_cols // elem_divisor  # valid structures for computation
    list_col = block_len // elem_divisor  # structures per block
    block_lens = list_col * 4  # structures per vmrgsort4 call
    block_lens_floats = block_len * 4  # floats per vmrgsort4 call

    repeat_times = valid_structs // block_lens  # vmrgsort4 call times (use valid_structs)

    # Generate random data only for valid portion (matching pto-isa which uses kTCols for computation)
    input_arr = np.random.uniform(low=0.0, high=1.0, size=(1, valid_structs)).astype(dtype)
    idx_arr = np.arange(valid_structs, dtype=np.uint32)

    # Step 1: Sort each block internally
    # Reshape to (total_blocks, list_col)
    input_reshaped = input_arr.reshape(-1, list_col)
    idx_reshaped = idx_arr.reshape(-1, list_col)

    # Sort each block descending
    sorted_indices = np.argsort(-input_reshaped, kind='stable', axis=1)
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)

    # Flatten back -> input0.bin (needs padding if cols > valid_structs)
    flat_input = sorted_input.flatten()
    flat_idx = sorted_idx.flatten()

    # Pad input with zeros if needed (for src_cols > valid_cols cases)
    if cols > valid_structs:
        pad_input = np.zeros(cols - valid_structs, dtype=dtype)
        pad_idx = np.zeros(cols - valid_structs, dtype=np.uint32)
        flat_input = np.concatenate((flat_input, pad_input))
        flat_idx = np.concatenate((flat_idx, pad_idx))

    # Step 2: Generate golden (globally sort each group, using valid_structs)
    # Take complete groups from valid portion
    input_group = flat_input[:valid_structs // block_lens * block_lens]
    idx_group = flat_idx[:valid_structs // block_lens * block_lens]

    # Reshape to (repeat_times, block_lens)
    single_output_reshape = input_group.reshape(-1, block_lens)
    single_idx_reshape = idx_group.reshape(-1, block_lens)

    # Globally sort each group descending
    single_sorted_indices = np.argsort(-single_output_reshape, kind='stable', axis=1)
    golden_values = np.take_along_axis(single_output_reshape, single_sorted_indices, axis=1).flatten()
    golden_indices = np.take_along_axis(single_idx_reshape, single_sorted_indices, axis=1).flatten()

    # Handle remaining elements from valid portion
    if valid_structs % block_lens != 0:
        zeros_output = np.zeros(valid_structs % block_lens, dtype=golden_values.dtype)
        zeros_index = np.zeros(valid_structs % block_lens, dtype=np.uint32)
        golden_values = np.concatenate((golden_values, zeros_output))
        golden_indices = np.concatenate((golden_indices, zeros_index))

    # Pad golden with zeros for full file size (cols > valid_structs)
    if cols > valid_structs:
        pad_output = np.zeros(cols - valid_structs, dtype=golden_values.dtype)
        pad_index = np.zeros(cols - valid_structs, dtype=np.uint32)
        golden_values = np.concatenate((golden_values, pad_output))
        golden_indices = np.concatenate((golden_indices, pad_index))

    os.makedirs(case["name"], exist_ok=True)
    with open(os.path.join(case["name"], "input0.bin"), 'wb') as f:
        for val, idx in zip(flat_input, flat_idx):
            write_value_index_pair(f, val, idx, dtype)

    with open(os.path.join(case["name"], "golden.bin"), 'wb') as f:
        for val, idx in zip(golden_values, golden_indices):
            write_value_index_pair(f, val, idx, dtype)

    print(f"[INFO] gen_data: {case['name']} src_cols={src_cols} valid_cols={valid_cols} "
          f"cols={cols} list_col={list_col} block_lens={block_lens} repeat_times={repeat_times}")


def gen_golden_multilist(case):
    """Generate golden data for Format2 (multi-list merge sort).

    Following pto-isa gen_data.py logic for multi-list:
    1. Generate sorted data for each input list (descending order)
    2. Concatenate all lists and globally sort (descending)
    3. Take top-k elements
    4. If exhausted=true, handle special termination logic

    Each input list is pre-sorted in descending order.
    Output is top-k merged sorted elements.
    """
    dtype = case["dtype"]
    list_num = case["list_num"]
    src_cols = case["src_cols"]  # structures per list
    topk = case["topk"]
    exhausted = case.get("exhausted", False)

    # Calculate actual cols (in elements) per src
    # Each structure = (value, index) pair = 8 bytes
    # For f32: 2 elements per structure (4 bytes value + 4 bytes index)
    # For f16: 4 elements per structure (2 bytes value + 2 bytes padding + 4 bytes index)
    elem_divisor = get_elem_divisor(dtype)

    # Generate sorted data for each input list
    output_arr_list = []
    output_idx_list = []
    last_data = []

    total_structures = sum(src_cols)

    for i in range(list_num):
        cols_i = src_cols[i]
        # Generate random data for this list
        input_arr = np.random.uniform(low=0.0, high=1.0, size=(1, cols_i)).astype(dtype)
        idx_arr = np.arange(cols_i, dtype=np.uint32).reshape(1, cols_i)  # Reshape to match input_arr

        # Sort in descending order
        sorted_indices = np.argsort(-input_arr, kind='stable', axis=1)
        sorted_input = np.take_along_axis(input_arr, sorted_indices, axis=1)
        sorted_idx = np.take_along_axis(idx_arr, sorted_indices, axis=1)

        # Flatten
        flat_input_i = sorted_input.flatten()
        flat_idx_i = sorted_idx.flatten()

        output_arr_list.append(flat_input_i)
        output_idx_list.append(flat_idx_i)

        # Track last element for exhausted case
        if cols_i > 0:
            last_data.append(flat_input_i[-1])
        else:
            last_data.append(0)

    # Concatenate and globally sort (descending)
    flat_input_group = np.concatenate(output_arr_list).flatten()
    flat_idx_group = np.concatenate(output_idx_list).flatten()

    sorted_indices_global = np.argsort(-flat_input_group, kind='stable')
    sorted_output_global = flat_input_group[sorted_indices_global]
    sorted_idx_global = flat_idx_group[sorted_indices_global]

    # Take top-k
    topk_sorted_output = sorted_output_global[:topk]
    topk_sorted_idx = sorted_idx_global[:topk]

    # Pad zeros if needed
    zeros_output = np.zeros(total_structures - topk, dtype=topk_sorted_output.dtype)
    zeros_index = np.zeros(total_structures - topk, dtype=np.uint32)
    topk_sorted_output_global = np.concatenate((topk_sorted_output, zeros_output))
    topk_sorted_idx_global = np.concatenate((topk_sorted_idx, zeros_index))

    if exhausted:
        handle_exhausted_list(list_num, topk_sorted_output_global, topk_sorted_idx_global, last_data)

    # Write input files (input0.bin, input1.bin, etc.)
    os.makedirs(case["name"], exist_ok=True)
    for i in range(list_num):
        input_file = os.path.join(case["name"], f"input{i}.bin")
        with open(input_file, 'wb') as f:
            for val, idx in zip(output_arr_list[i], output_idx_list[i]):
                write_value_index_pair(f, val, idx, dtype)

    # Write golden output file
    with open(os.path.join(case["name"], "golden.bin"), 'wb') as f:
        for val, idx in zip(topk_sorted_output_global, topk_sorted_idx_global):
            write_value_index_pair(f, val, idx, dtype)

    print(f"[INFO] gen_data: {case['name']} list_num={list_num} "
          f"src_cols={src_cols} total_structures={total_structures} topk={topk} exhausted={exhausted}")


def gen_golden_topk(case):
    """Generate golden data for TopK (full iterative merge).

    Following pto-isa RunTMrgsortTopk logic:
    1. Generate unsorted raw data -> input0.bin
    2. Initial: sort each block internally
    3. Iterative merge loop: blockLen *= 4
       - Each iteration: Format1 merge (4 blocks -> 1)
       - Copy result back for next iteration
    4. Final: take top-k from globally sorted data -> golden.bin

    This matches the full TopK template implementation.
    """
    dtype = case["dtype"]
    src_shape = _to_tuple(case["src_shape"])
    dst_shape = _to_tuple(case["dst_shape"])
    valid_shape = _to_tuple(case["valid_shape"])
    topk = case["topk"]  # output structures count
    block_len = case["block_len"]

    src_rows, src_cols = src_shape
    valid_rows, valid_cols = valid_shape

    # Get element divisor based on dtype
    elem_divisor = get_elem_divisor(dtype)

    # Structure units (following pto-isa)
    cols = valid_cols // elem_divisor  # total structures
    list_col = block_len // elem_divisor  # structures per block

    # Generate unsorted raw data
    input_arr = np.random.uniform(low=0.0, high=1.0, size=(1, cols)).astype(dtype)
    idx_arr = np.arange(cols, dtype=np.uint32)

    # Step 1: Sort each block internally (Format1 preparation)
    input_reshaped = input_arr.reshape(-1, list_col)
    idx_reshaped = idx_arr.reshape(-1, list_col)

    sorted_indices = np.argsort(-input_reshaped, kind='stable', axis=1)
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)

    # Flatten -> input0.bin (block-wise sorted)
    flat_input = sorted_input.flatten()
    flat_idx = sorted_idx.flatten()

    # Step 2: Iterative merge (blockLen *= 4 loop)
    current_data = flat_input.copy()
    current_idx = flat_idx.copy()
    current_block_len = list_col  # structures per block

    iteration = 1
    while current_block_len * 4 <= cols:
        # Format1 merge at this block length
        # Merge groups of 4 blocks into 1
        block_lens = current_block_len * 4  # structures per merge group
        num_groups = cols // block_lens

        # Process each group
        for g in range(num_groups):
            start = g * block_lens
            end = start + block_lens
            group_vals = current_data[start:end]
            group_idx = current_idx[start:end]

            # Sort this group descending
            sort_indices = np.argsort(-group_vals, kind='stable')
            current_data[start:end] = group_vals[sort_indices]
            current_idx[start:end] = group_idx[sort_indices]

        # Update block length for next iteration
        current_block_len = current_block_len * 4
        iteration += 1

    # Step 3: Handle tail blocks (if current_block_len < cols)
    # Simplified: just globally sort the remaining data
    if current_block_len < cols:
        # Global sort for tail handling
        sort_indices = np.argsort(-current_data, kind='stable')
        current_data = current_data[sort_indices]
        current_idx = current_idx[sort_indices]

    # Step 4: Take top-k
    golden_values = current_data[:topk]
    golden_indices = current_idx[:topk]

    # Write files
    os.makedirs(case["name"], exist_ok=True)
    with open(os.path.join(case["name"], "input0.bin"), 'wb') as f:
        for val, idx in zip(flat_input, flat_idx):
            write_value_index_pair(f, val, idx, dtype)

    # Pad zeros if needed (to match dst capacity)
    dst_structures = dst_shape[1] // elem_divisor
    zeros_values = np.zeros(dst_structures - topk, dtype=golden_values.dtype)
    zeros_indices = np.zeros(dst_structures - topk, dtype=np.uint32)
    golden_values_padded = np.concatenate((golden_values, zeros_values))
    golden_indices_padded = np.concatenate((golden_indices, zeros_indices))

    with open(os.path.join(case["name"], "golden.bin"), 'wb') as f:
        for val, idx in zip(golden_values_padded, golden_indices_padded):
            write_value_index_pair(f, val, idx, dtype)

    print(f"[INFO] gen_data: {case['name']} src_cols={src_cols} valid_cols={valid_cols} "
          f"cols={cols} structures topk={topk} structures block_len={block_len} "
          f"iterations={iteration}")


def gen_golden_data():
    """Generate golden data for all cases."""
    for case in CASES:
        setup_case_rng(case)

        format_type = case.get("format", "single")
        if format_type == "single":
            gen_golden_single(case)
        elif format_type == "multi":
            gen_golden_multilist(case)
        elif format_type == "topk":
            gen_golden_topk(case)
        else:
            print(f"[WARN] Unsupported format: {format_type} for case {case['name']}")


if __name__ == "__main__":
    gen_golden_data()
