#include "fm_scan.cuh"
// a dummy place holder used for debugging with nvcc compiler

// when compiling with nvcc
// -I with print(torch.utils.cpp_extension.include_paths())
// -I/path/to/cuda/include (/usr/local/cuda/include)

// nvcc -O3 -std=c++17 -lineinfo --ptxas-options=-v -I .... -o fm_scan.o -c fm_scan.cu

// nvcc -O3 -std=c++17 -lineinfo --ptxas-options=-v \ 
// -I$(python -c 'import torch; print(torch.utils.cpp_extension.include_paths()[0])') \ this line needs to enumerate all the list
// -I$(python -c 'from sysconfig import get_paths as gp; print(gp()["include"])') \ 
// -I/usr/local/cuda/include -c fm_scan.cu -o fm_scan.o
