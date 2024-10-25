#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <vector>

struct CudaReduction
{
	std::vector<unsigned int> grid_v;
	std::vector<unsigned int> N_v;

	unsigned int steps = 0, threads = 1024;

	double* res_array = nullptr;
	double res = 0;
	double** arr = nullptr;

	void (*reduction_kernel)();
	enum ReductionType	{ABSSUM, SIGNEDSUM,	DOTPRODUCT 	} type = ABSSUM;

	CudaReduction(double* device_ptr, unsigned int N, unsigned int thr = 1024);
	CudaReduction(unsigned int N, unsigned int thr = 1024);
	CudaReduction();
	~CudaReduction();
	void set_reduced_size(unsigned int N, unsigned int thr, bool doubleRead);


	void print_check();
	double reduce(bool withCopy = true);
	double reduce(double* device_ptr, bool withCopy = true);
	static double reduce(double* device_ptr, unsigned int N, unsigned int thr = 1024, bool withCopy = true);



	void auto_test();
};
