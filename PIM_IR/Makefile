all: build

cmake:
	mkdir -p build && cd build && cmake -G Ninja ..;

build: cmake
	cd build && cmake --build . --target PIM-IR

clean_build: cmake
	cd build && cmake --build . --target PIM-IR

clean:
	rm -rf build
