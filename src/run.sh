#!/bin/bash

rm -rf ./main

g++ main.cpp matrix.cpp multiply.cpp -std=c++1z -pthread -mfma -o main -D JUDGE_RIGHT -D N=128 -D M=128 -D P=128
./main

rm -rf ./main

g++ main.cpp matrix.cpp multiply.cpp -std=c++1z -pthread -mfma -g -o main -D N=512 -D M=512 -D P=512
./main