#include "CudaKernels.cuh"
#include "stdafx.h"
#include "SequentialSvm.h"

///<summary>Used to end the training cycle.
///</summary>
__host__ __device__ double gaussKernel(const double* a, int aI, const double* b, const int bI, const int width, const double g)
{
	int aIw = aI * width;
	int bIw = bI * width;
	double product=1;
	double innerSum = 0;
	for (int j = 0; j < width; ++j)
	{
		product = a[aIw + j] - b[bIw + j];
		product *= product;
		innerSum += product;
	}
	return exp(-g*innerSum);
}

__host__ __device__ double calcAlpha(double* alpha, const double sum, const double* y, double* step, const double C, int idx)
{
	auto newAlpha = alpha[idx] + step[idx] - step[idx] * y[idx] * sum;
	if (newAlpha > C)
		newAlpha = C;
	else if (newAlpha < 0)
		newAlpha = 0.0;
	return newAlpha;
}

__global__ void classificationKernel(double *saida, const double *tX, const double *tY, const double *vX, const double *alpha, const double g, const int index, const int width, const int max)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > max) return;
	saida[idx] = tY[idx] * alpha[idx] * gaussKernel(tX, idx, vX, index, width, g);
}

__global__ void trainingKernelLoop(double *sum, const double *alpha, const double *x, const double *y, const double g, const int width, const int height, const int batchStart, const int batchEnd)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > height) return;
	double outerSum = sum[idx];
	for (int i = batchStart; i < batchEnd; i++){
		outerSum += alpha[i] * y[i] * gaussKernel(x, idx, x, i, width, g);
	}
	sum[idx] = outerSum;
}

__global__ void trainingKernelFinish(double *alpha, double *sum, const double *y, const int nSamples, double *step, double *last, const double C)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > nSamples) return;
	double newAlpha = calcAlpha(alpha, sum[idx], y, step, C, idx);

	sum[idx] = 0.0;

	auto dif = newAlpha - alpha[idx];
	if (dif*last[idx] < 0)
		step[idx] /= 2;
	last[idx] = dif;
	alpha[idx] = newAlpha;
}

__global__ void initArray(double *array, const double value, const int max)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < max)
		array[idx] = value;
}