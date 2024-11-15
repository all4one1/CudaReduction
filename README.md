# CudaReduction

## Usage:
```cpp
#include "CuReduction.h"

int main()
{
    double* ptr_d; // your device pointer
    int N = 123456;
    int B = N * sizeof(double);
    cudaMalloc((void**)&ptr_d, B);

    {   // usage 1
        CudaReduction CuRe(ptr_d, N, 512);
        double sum = CuRe.reduce();
        CuRe.auto_test();
    }

    {   // usage 2
        double sum = CudaReduction::reduce(ptr_d, N, 512);
    }

    {   // usage 3
        CudaReduction CuRe(N, 512);
        double sum = CuRe.reduce(ptr_d);
    }

    {
        // usage 4
        // via CUDA Graph
        CudaReduction CuRe(ptr_d, N, 512);
        CuGraph gr = CuRe.make_graph(ptr_d, true);
        gr.instantiate();
        gr.launch();
    }

    return 0;
}
