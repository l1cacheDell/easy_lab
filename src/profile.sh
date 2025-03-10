#!/bin/bash

rm -rf ./main

g++ main.cpp matrix.cpp multiply.cpp -std=c++1z -pthread -mfma -g -o main -D N=1024 -D M=1024 -D P=1024
perf stat -e cycles,instructions,cache-misses,branch-misses,L1-dcache-load-misses,L1-dcache-loads ./main