# Clang-based frontend to convert PIM-API to MLIR 

1. Set up Polygeist [[1]](#1) with patch
```
  git submodule update --init --recursive
  cd Polygeist
  git am ../pim-avail.patch
```
2. Build LLVM, MLIR, Clang
```
mkdir Polygeist/llvm-project/build && cd Polygeist/llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir  # can be skipped
```
3. Build necessary PIM-IR libraries
```
mkdir PIM_IR/build && cd PIM_IR/build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../../Polygeist/llvm-project/build/lib/cmake/mlir
cmake --build . --target CallToPIM
```
4. Build mlir-clang with PIM-IR
```
mkdir Polygeist/build && cd Polygeist/build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
cmake --build . --target mlir-clang
```


To test,
```
./Polygeist/build/mlir-clang/mlir-clang --raise-scf-to-affine --pim-avail -S --function=cblas_saxpy PIM_BLAS/src/cblas_saxpy.c
```

## References
<a id="1">[1]</a> 
Moses, William S., et al. "Polygeist: Affine C in MLIR." (2021).
