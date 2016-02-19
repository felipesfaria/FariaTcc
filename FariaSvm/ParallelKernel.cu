#include "stdafx.h"
#include "ParallelKernel.cuh"
#include "DataSet.h"
#include "cuda_runtime.h"

__global__ void addKernel(double *c, const double *a, const double *b, const double *gama)
{
	int i = threadIdx.x;
	c[i] = a[i] - b[i];
	c[i] *= c[i];
}

__global__ void map(double *saida, const double *x, const int posI, const int posJ)
{
	int i = threadIdx.x;
	saida[i] = x[posI + i] - x[posJ + i];
	saida[i] *= saida[i];
}

ParallelKernel::ParallelKernel()
{
	_type = NONE;
}


ParallelKernel::~ParallelKernel()
{
}

double *dev_x;
double *dev_s;
double *c;
int size;
void ParallelKernel::Init(DataSet ds)
{
	_type = ds.kernelType;
	cudaError_t cudaStatus;
	switch (_type)
	{
	case GAUSSIAN:
		_sigma = ds.Gama;
		break;
	default:
		throw(new std::exception("Not Implemented exception"));
	}
	int completeSize = ds.X.size()*ds.nFeatures;
	size = ds.nFeatures;
	double* x = (double*)malloc(sizeof(double)*completeSize);
	c = (double*)malloc(sizeof(double)*size);
	int k = 0;
	for (int i = 0; i < ds.X.size(); i++){
		for (int j = 0; k < size; j++){
			x[k++] = ds.X[i][j];
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
	cudaStatus = cudaMalloc((void**)&dev_s, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(new exception("cudaMalloc failed!"));
	}

	cudaStatus = cudaMemcpy(dev_x, x, completeSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(new exception("cudaMemcpy failed!"));
	}
}

double ParallelKernel::K(std::vector<double> x, std::vector<double> y)
{
	double *a = (double*)malloc(sizeof(double)*x.size());
	double *b = (double*)malloc(sizeof(double)*x.size());
	double *c = (double*)malloc(sizeof(double)*x.size());
	for (int i = 0; i < x.size(); i++){
		a[i] = x[i];
		b[i] = y[i];
	}
	cudaError_t cudaStatus = AddWithCuda(c, a, b, x.size());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	double sum = 0;
	for (int i = 0; i < x.size(); i++){
		sum += c[i];
	}
	double gama = 1 / (2 * _sigma*_sigma);
	return exp(-gama*sum);
}

double ParallelKernel::K(int i, int j)
{
	cudaError_t cudaStatus;
	map <<<1, size >>>(dev_s, dev_x, i*size, j*size);

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
	cudaStatus = cudaMemcpy(c, dev_s, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(new exception("cudaMemcpy failed!"));
	}
	double sum = 0;
	for (int i = 0; i < size; i++){
		sum += c[i];
	}
	double gama = 1 / (2 * _sigma*_sigma);
	return exp(-gama*sum);
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