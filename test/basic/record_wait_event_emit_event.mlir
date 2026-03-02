// RUN: ptoas --emit-manual-sync-as-event %s | FileCheck %s

module {
  func.func @sync_ops() {
    pto.record_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    pto.wait_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    return
  }
}

// CHECK: #include "pto/pto-inst.hpp"
// CHECK: __global__ AICORE void sync_ops()
// CHECK: Event<Op::TLOAD, Op::VECTOR>
// CHECK: .Record()
// CHECK: .Wait()
// CHECK-NOT: PTOAS__MANUAL_EVENT_WAIT
// CHECK-NOT: PTOAS__MANUAL_EVENT_RECORD
// CHECK-NOT: TSYNC
// CHECK-NOT: set_flag
// CHECK-NOT: wait_flag
