#/bin/bash

rm -rf build
mkdir build
cd build
cmake .. \
	-DCMAKE_BUILD_TYPE=Release \
	-DPYBIND11_PATH=/path/to/anaconda3/envs/yolov8/lib/python3.9/site-packages/pybind11/share/cmake/pybind11
make -j2
make install
