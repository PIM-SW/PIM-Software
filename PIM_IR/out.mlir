module  {
  func @boo(%arg0: i32, %arg1: f32, %arg2: memref<100xf32>, %arg3: memref<100xf32>, %arg4: tensor<1xf64>) -> tensor<1xf64> {
    %0 = pnm.vector(0 %arg2 %arg3 16)(memref<100xf32> memref<100xf32> memref<100xf32>)
    %1 = pnm.vector(0 %arg3 %0 16)(memref<100xf32> memref<100xf32> memref<100xf32>)
    %2 = pnm.vector(0 %0 %1 16)(memref<100xf32> memref<100xf32> memref<100xf32>)
    %3 = pnm.vector(0 %2 %1 16)(memref<100xf32> memref<100xf32> memref<100xf32>)
    pim.return %arg4 : tensor<1xf64>
  }
}

