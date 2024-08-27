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

    {   // usage 1
        CudaReduction CuRe(ptr_d, N, 1024);
        double sum = CuRe.reduce();
        CuRe.auto_test();
    }

    {   // usage 2
        double sum = CudaReduction::reduce(ptr_d, N, 1024);
    }

    {   // usage 3
        CudaReduction CuRe(N, 1024);
        double sum = CuRe.reduce(ptr_d);
    }

   

    return 0;
}