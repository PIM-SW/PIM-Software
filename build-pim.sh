mkdir PIM_IR/build && cd PIM_IR/build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../../Polygeist/llvm-project/build/lib/cmake/mlir
cmake --build . --target CallToPIM
cd -
