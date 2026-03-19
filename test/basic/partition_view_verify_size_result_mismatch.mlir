// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @partition_view_verify_size_result_mismatch(%ptr : !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index

    %tv = pto.make_tensor_view %ptr, shape = [%c8, %c64], strides = [%c64, %c1] : !pto.tensor_view<8x64xf32>
    %pv = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c4, %c16] : !pto.tensor_view<8x64xf32> -> !pto.partition_tensor_view<4x64xf32>
    return
  }
}

// CHECK: error: 'pto.partition_view' op size/result mismatch at dim 1
