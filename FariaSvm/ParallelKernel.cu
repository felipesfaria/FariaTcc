#include "stdafx.h"
#include "ParallelKernel.cuh"
#include "DataSet.h"
#include "Logger.h"
#include "cuda_runtime.h"

__global__ void addKernel(double *c, const double *a, const double *b, const double *gama)
{
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
	c[i] *= c[i];
}

__global__ void myFunc(double *saida, const double *x, const int *posI, const int *posJ)
{
	int i = threadIdx.x;
	saida[i] = x[posI[0] + i] - x[posJ[0] + i];
	saida[i] *= saida[i];
}

ParallelKernel::ParallelKernel(const DataSet& ds)
{
	Logger::Stats("Kernel:", "Parallel");
	_type = ds.kernelType;
	cudaError_t cudaStatus;
	switch (_type)
	{
	case GAUSSIAN:
		_sigma = 1 / (2 * ds.Gama*ds.Gama);
		break;
	default:
		throw(new std::exception("Not Implemented exception"));
	}
	int completeSize = ds.X.size()*ds.nFeatures;
	features = ds.nFeatures;
	double* hst_x = (double*)malloc(sizeof(double)*completeSize);
	hst_s = (double*)malloc(sizeof(double)*features);
	hst_i = (int*)malloc(sizeof(int));
	hst_j = (int*)malloc(sizeof(int));
	int k = 0;
	for (int i = 0; i < ds.X.size(); i++){
		for (int j = 0; j < features; j++){
			auto v = ds.X[i][j];
			hst_x[k++] = v;
		}
	}
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		throw(new exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
	}

	cudaStatus = cudaMalloc((void**)&dev_x, completeSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(new exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_s, features * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(new exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_i, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(new exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_j, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(new exception("cudaMalloc failed!"));
	}

	cudaStatus = cudaMemcpy(dev_x, hst_x, completeSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(new exception("cudaMemcpy failed!"));
	}
	free(hst_x);
}


ParallelKernel::~ParallelKernel()
{
	cudaFree(dev_x);
	cudaFree(dev_s);
	free(hst_s);
	cudaFree(dev_i);
	free(hst_i);
	cudaFree(dev_j);
	free(hst_j);
}

double ParallelKernel::K(int i, int j, const DataSet& ds)
{
	cudaError_t cudaStatus;
	hst_i[0] = i*features;
	hst_j[0] = j*features;

	cudaStatus = cudaMemcpy(dev_i, hst_i, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(new exception("cudaMemcpy failed!"));
	}
	cudaStatus = cudaMemcpy(dev_j, hst_j, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(new exception("cudaMemcpy failed!"));
	}

	myFunc << <1, features >> >(dev_s, dev_x, dev_i, dev_j);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		throw(new exception("addKernel launch failed: %s\n"));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		throw(new exception("cudaDeviceSynchronize returned error code %d after launching addKernel!\n"));
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hst_s, dev_s, features * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(new exception("cudaMemcpy failed!"));
	}
	double sum = 0;
	for (int i = 0; i < features; i++){
		sum += hst_s[i];
	}
	return exp(-_sigma*sum);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t ParallelKernel::AddWithCuda(double *c, const double *a, const double *b, unsigned int size)
{
	double *gama = (double*)malloc(sizeof(double));
	*gama = 1 / (2 * _sigma*_sigma);
	double *dev_a = 0;
	double *dev_b = 0;
	double *dev_c = 0;
	double *dev_g = 0;
	cudaError_t cudaStatus;
	try{

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw(new exception("cudaMalloc failed!"));
		}
		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw(new exception("cudaMalloc failed!"));
		}
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw(new exception("cudaMalloc failed!"));
		}
		cudaStatus = cudaMalloc((void**)&dev_g, sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw(new exception("cudaMalloc failed!"));
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw(new exception("cudaMemcpy failed!"));
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw(new exception("cudaMemcpy failed!"));
		}

		cudaStatus = cudaMemcpy(dev_g, gama, sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw(new exception("cudaMemcpy failed!"));
		}

		// Launch a kernel on the GPU with one thread for each element.
		addKernel <<<1, size >>>(dev_c, dev_a, dev_b, dev_g);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			throw(new exception("addKernel launch failed: %s\n"));
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			throw(new exception("cudaDeviceSynchronize returned error code %d after launching addKernel!\n"));
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw(new exception("cudaMemcpy failed!"));
		}
	}
	catch (exception &e){
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
	}
	return cudaStatus;
}