#include "CuReduction.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
//#include "nvidia_kernels.h"
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void init_test(double* data, unsigned int n)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		data[i] = i + 1;
	}
}
__global__ void gpu_print(double* f)
{
	printf("message: %f", f[0]);
	printf("\n");
}
__global__ void reduction_abs_sum(double* data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		shared[tid] = abs(data[i]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}
__global__ void reduction_signed_sum(double* data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		shared[tid] = data[i];
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}
}
__global__ void dot_product(double* v1, double *v2, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) 
	{
		shared[tid] = v1[i] * v2[i];
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}
}


template <class T, unsigned int blockSize>
__global__ void reduce5abs(T* g_idata, T* g_odata, unsigned int n) {
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ T sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

	T mySum = (i < n) ? g_idata[i] : 0;

	if (i + blockSize < n) mySum += g_idata[i + blockSize];

	sdata[tid] = mySum;
	cg::sync(cta);

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256)) {
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128)) {
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64)) {
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32) {
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
			mySum += tile32.shfl_down(mySum, offset);
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}



CudaReduction::CudaReduction()
{

}

CudaReduction::CudaReduction(unsigned int N, unsigned int thr)
{
	set_reduced_size(N, thr, true);

	if (res_array != nullptr) cudaFree(res_array);
	cudaMalloc((void**)&res_array, sizeof(double) * N_v[1]);

	if (arr != nullptr) delete[] arr;
	arr = new double* [steps + 1];
}

CudaReduction::CudaReduction(double* device_ptr, unsigned int N, unsigned int thr)
{
	set_reduced_size(N, thr, true);

	if (res_array != nullptr) cudaFree(res_array);
	cudaMalloc((void**)&res_array, sizeof(double) * N_v[1]);

	if (arr != nullptr) delete[] arr;
	arr = new double* [steps + 1];

	arr[0] = device_ptr;
	for (unsigned int i = 1; i <= steps; i++)
		arr[i] = res_array;
}


CudaReduction::~CudaReduction()
{
	cudaFree(res_array); res_array = nullptr;
	delete[] arr; arr = nullptr;
	grid_v.clear();
	N_v.clear();
}

void CudaReduction::set_reduced_size(unsigned int N, unsigned int thr, bool doubleRead)
{
	if (thr < 64)
	{
		std::cout << "more threads needed " << std::endl;
		threads = 64;
	}

	unsigned int temp_ = N;
	threads = thr;
	N_v.push_back(N);

	if (threads == 512) reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 512>);
	if (threads == 256) reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 256>);
	if (threads == 128) reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 128>);
	if (threads == 64)  reduction_kernel = reinterpret_cast<void*>(&reduce5abs<double, 64>);


	steps = 0;
	while (true)
	{
		steps++;
		if (doubleRead) temp_ = (temp_ + (threads * 2 - 1)) / (threads * 2);
		else temp_ = (temp_ + threads - 1) / threads;

		grid_v.push_back(temp_);
		N_v.push_back(temp_);
		if (temp_ == 1)  break;
	}
}

void CudaReduction::print_check()
{
	gpu_print << <1, 1 >> > (res_array);
}

void CudaReduction::auto_test()
{
	double* ptr_d;
	int N = 123456;

	cudaMalloc((void**)&ptr_d, N * sizeof(double));
	init_test << <1024, 1024 >> > (ptr_d, N);

	std::cout << std::fixed;
	std::cout << "Exact value = " << N / 2.0 * (N + 1)  << std::endl;
	std::cout << "Cuda result = " << CudaReduction::reduce(ptr_d, N, 128) << std::endl;
}


double CudaReduction::check_on_cpu(double* device_ptr, unsigned int N)
{
	double* f = new double[N];
	cudaMemcpy(f, device_ptr, sizeof(double) * N, cudaMemcpyDeviceToHost);
	double s = 0;
	for (unsigned int i = 0; i < N; i++)
		s += abs(f[i]);
	return s;
}


double CudaReduction::reduce_legacy(bool withCopy)
{
	switch (type)
	{
	case CudaReduction::ABSSUM:	
		for (unsigned int i = 0; i < steps; i++)
			reduction_abs_sum << < grid_v[i], threads, 1024 * sizeof(double) >> > (arr[i], N_v[i], arr[i + 1]);
		break;
	case CudaReduction::SIGNEDSUM:
		for (unsigned int i = 0; i < steps; i++)
			reduction_signed_sum << < grid_v[i], threads, 1024 * sizeof(double) >> > (arr[i], N_v[i], arr[i + 1]);
		break;
	case CudaReduction::DOTPRODUCT:
		// todo
		//for (unsigned int i = 0; i < steps; i++)
		//	dot_product << < Gp[i], threads, 1024 * sizeof(double) >> > (arr[i], arr[i], Np[i], arr[i + 1]); 
		break;
	default:
		break;
	}

	if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

	return res;
}

double CudaReduction::reduce(double* device_ptr, bool withCopy)
{
	arr[0] = device_ptr;
	for (unsigned int i = 1; i <= steps; i++)
		arr[i] = res_array;

	return reduce(withCopy);
}

double CudaReduction::reduce(double* device_ptr, unsigned int N, unsigned int thr, bool withCopy)
{
	CudaReduction temp(device_ptr, N, thr);
	return temp.reduce(withCopy);
}


double CudaReduction::reduce(bool withCopy)
{
	//for (unsigned int i = 0; i < steps; i++) reduce_<double>(N_v[i], threads, grid_v[i], 5, arr[i], arr[i + 1]);

	//for (unsigned int i = 0; i < steps; i++)
	//{
	//	void* args[] = { &arr[i], &arr[i + 1], &N_v[i] };
	//	cudaLaunchKernel(reduction_kernel, grid_v[i], threads, args, smem, 0);
	//}

	if (threads == 512) for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 512> <<< grid_v[i], threads, smem >>> (arr[i], arr[i + 1], N_v[i]);
	if (threads == 256) for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 256> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);
	if (threads == 128) for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 128> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);
	if (threads == 64)  for (unsigned int i = 0; i < steps; i++) reduce5abs<double, 64> << < grid_v[i], threads, smem >> > (arr[i], arr[i + 1], N_v[i]);

	if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);
	return res;
}


CuGraph CudaReduction::make_graph(double* device_ptr, bool withCopy)
{
	arr[0] = device_ptr;
	for (unsigned int i = 1; i <= steps; i++)
		arr[i] = res_array;

	CuGraph graph;
	for (unsigned int i = 0; i < steps; i++)
	{
		void* args[3] = { &arr[i], &N_v[i], &arr[i + 1] };
		graph.add_kernel_node(threads, grid_v[i], reduction_abs_sum, args, smem);
	}

	if (withCopy) graph.add_copy_node(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

	return graph;
}



