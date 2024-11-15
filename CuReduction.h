#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <vector>

#include "CuGraph.h"


struct CudaReduction
{
	std::vector<unsigned int> grid_v;
	std::vector<unsigned int> N_v;

	#define def_threads 512
	unsigned int N = 0;
	unsigned int steps = 0, threads = def_threads, smem = sizeof(double) * def_threads;

	double* res_array = nullptr;
	double res = 0;
	double** arr = nullptr;

	void (*reduction_kernel)();
	enum ReductionType	{ABSSUM, SIGNEDSUM,	DOTPRODUCT 	} type = ABSSUM;

	CudaReduction(double* device_ptr, unsigned int N, unsigned int thr = def_threads);
	CudaReduction(unsigned int N, unsigned int thr = 1024);
	CudaReduction();
	~CudaReduction();
	void set_reduced_size(unsigned int N, unsigned int thr, bool doubleRead);


	void print_check();
	double reduce(bool withCopy = true);
	double reduce_legacy(bool withCopy = true);
	double reduce(double* device_ptr, bool withCopy = true);
	static double reduce(double* device_ptr, unsigned int N, unsigned int thr = def_threads, bool withCopy = true);

	double check_on_cpu(double* device_ptr, unsigned int N);

	void auto_test();

    CuGraph CudaReduction::make_graph(double* device_ptr, bool withCopy);
};
