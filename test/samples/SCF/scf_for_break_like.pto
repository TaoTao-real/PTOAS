// RUN: ptoas %s | FileCheck %s

module {
  func.func @for_break_like_kernel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    %false = arith.constant false
    %one = arith.constant 1.0 : f32
    %ten = arith.constant 10.0 : f32

    %tile = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %final_alive = scf.for %i = %c0 to %c4 step %c1
        iter_args(%alive = %true) -> (i1) {
      %next_alive = scf.if %alive -> (i1) {
        pto.tadds ins(%tile, %one
          : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                          blayout=row_major, slayout=none_box, fractal=512, pad=0>,
            f32)
          outs(%tile
          : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                          blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        %break_now = arith.cmpi eq, %i, %c2 : index
        %alive_after = scf.if %break_now -> (i1) {
          scf.yield %false : i1
        } else {
          pto.tadds ins(%tile, %ten
            : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                            blayout=row_major, slayout=none_box, fractal=512, pad=0>,
              f32)
            outs(%tile
            : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                            blayout=row_major, slayout=none_box, fractal=512, pad=0>)
          scf.yield %true : i1
        }
        scf.yield %alive_after : i1
      } else {
        scf.yield %alive : i1
      }
      scf.yield %next_alive : i1
    }

    %stopped = arith.xori %final_alive, %true : i1
    scf.if %stopped {
      pto.tadds ins(%tile, %one
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>,
          f32)
        outs(%tile
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    }
    return
  }
}

// CHECK: for (
// CHECK: if (
// CHECK: TADDS(
// CHECK: if (
// CHECK: } else {
// CHECK: TADDS(
