#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include "CuReduction.h"

using namespace std;

int main()
{

    double* ptr_d;
    int N = 123456;
    int B = N * sizeof(double);
    cudaMalloc((void**)&ptr_d, B);


    // usage 1
    CudaReduction CuRe(ptr_d, N, 1024);
    double sum1 = CuRe.reduce();

    // usage 2
    double sum2 = CudaReduction::reduce(ptr_d, N, 1024);


    CuRe.auto_test();

    return 0;
}