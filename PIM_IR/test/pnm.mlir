module  {
  func @boo(%arg0: i32, %arg1: f32, %arg2: memref<100xf32>, %arg3: memref<100xf32>, %arg4: tensor<1xf64>) -> tensor<1xf64> {
    %4 = "pim.simd_add"(%arg2,  %arg3) : (memref<100xf32>, memref<100xf32>) -> memref<100xf32>
    %7 = "pim.simd_add"(%arg3,  %4) : (memref<100xf32>, memref<100xf32>) -> memref<100xf32>
    %12 = "pim.simd_add"(%4,  %7) : (memref<100xf32>, memref<100xf32>) -> memref<100xf32>
    %16 = "pim.simd_add"(%12, %7) : (memref<100xf32>, memref<100xf32>) -> memref<100xf32>
    pim.return %arg4 : tensor<1xf64>
  }
}

