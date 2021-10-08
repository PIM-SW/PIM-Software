# An out-of-tree MLIR dialect for supporting PIM

## Building

This setup assumes that you have downloaded and built LLVM and MLIR at your home directory. 

LLVM_DIR: ~/llvm-project/build/lib/cmake/llvm
MLIR_DIR: ~/heelim/llvm-project/build/lib/cmake/mlir

To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja ..
cmake --build . --target PIM-IR
```
or just type
```sh
make
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

after build, test with following command
```sh
./build/PIM-IR/PIM-IR test/pnm.mlir --convert-pim
```

if you want instruction format, 
```sh
./build/PIM-IR/PIM-IR test/pnm.mlir --convert-pim > out.mlir
python3 translate.py 
```

