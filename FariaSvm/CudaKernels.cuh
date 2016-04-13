#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

__host__ __device__ double calcAlpha(double* alpha, const double sum, const double* y, double* step, const double C, int idx);

__host__ __device__ double gaussKernel(const double* a, int aIndex, const double* b, const int bIndex, const int width, const double g);

__global__ void classificationKernel(double *saida, const double *tX, const double *tY, const double *vX, const double *alpha, const double gama, const int index, const int width, const int max);

__global__ void trainingKernelLoop(double *sum, const double *alpha, const double *x, const double *y, const double gama, const int nFeatures, const int nSamples, const int batchStart, const int batchEnd);

__global__ void trainingKernelFinish(double *alpha, double *sum, const double *y, const int nSamples, double *step, double *last, const double C);

__global__ void initArray(double *array, const double value);