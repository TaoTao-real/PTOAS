// RUN: ptoas --pto-arch a5 --enable-insert-sync %s | FileCheck %s

module attributes {"pto.device-spec" = "Ascend950"} {
  // A5: vec->mat TINSERT must use PIPE_MTE3 (custom UB->L1 path).
  func.func @tinsert_vec_mat_pipeline(%a: memref<32x32xf16, #pto.address_space<gm>>,
                                      %b: memref<32x32xf16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %src_vec = memref.alloc() : memref<32x32xf16, #pto.address_space<vec>>
    %dst_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %tmp_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %out_left = memref.alloc() : memref<32x32xf16, #pto.address_space<left>>

    pto.tload ins(%a : memref<32x32xf16, #pto.address_space<gm>>)
              outs(%src_vec : memref<32x32xf16, #pto.address_space<vec>>) {layout = #pto.layout<nd>}
    pto.tinsert ins(%src_vec, %c0, %c0 : memref<32x32xf16, #pto.address_space<vec>>, index, index)
               outs(%dst_mat : memref<32x32xf16, #pto.address_space<mat>>)
    pto.tload ins(%b : memref<32x32xf16, #pto.address_space<gm>>)
              outs(%tmp_mat : memref<32x32xf16, #pto.address_space<mat>>) {layout = #pto.layout<nd>}
    pto.tmov ins(%dst_mat : memref<32x32xf16, #pto.address_space<mat>>)
             outs(%out_left : memref<32x32xf16, #pto.address_space<left>>)
    return
  }
}

// CHECK-LABEL: __global__ AICORE void tinsert_vec_mat_pipeline(
// CHECK: set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
// CHECK: wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
// CHECK: TINSERT(
// CHECK: set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
// CHECK: wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
