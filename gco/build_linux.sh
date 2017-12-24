#!/bin/bash

mkdir -p build
cd build

cmake -G"Unix Makefiles" -D CMAKE_BUILD_TYPE="Release" ..

make -j4