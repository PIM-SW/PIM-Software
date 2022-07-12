git submodule update --init --recursive
cd Polygeist && git am ../pim-avail.patch && cd -
mkdir Polygeist/llvm-project/build && cd Polygeist/llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
cd -
