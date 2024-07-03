#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

struct CudaReduction
{
	//GPU-Grid points and Node (total) points
	unsigned int* Gp, * Np;
	unsigned int steps = 0, threads = 1024;

	double* res_array;
	double res = 0;
	double** arr;

	CudaReduction(double* device_ptr, unsigned int N, unsigned int thr = 1024);
	CudaReduction();

	void print_check();
	double reduce();
	static double reduce(double* device_ptr, unsigned int N, unsigned int thr = 1024)
	{
		CudaReduction temp(device_ptr, N, thr);
		return temp.reduce();
	}

	void auto_test();
};
