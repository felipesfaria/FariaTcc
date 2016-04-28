#pragma once
#include "stdafx.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

using namespace std;
#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__,  __LINE__,cudaGetErrorString(err)); \
      throw exception(cudaGetErrorString(err)); } }

//__host__ __device__ double calcAlphaMultiStep(double* alpha, const double sum, const double* y, double* step, const double C, int idx);
//
//__host__ __device__ double calcAlphaSingleStep(double* alpha, const double sum, const double* y, double step, const double C, int idx);

__host__ __device__ double calcAlpha(double alpha, const double sum, const double y, double step, const double C);

__host__ __device__  void updateStep(double& step, double& oldDif, double newDif);

__host__ __device__ double gaussKernel(const double* a, int aIndex, const double* b, const int bIndex, const int width, const double g);

__global__ void classificationKernel(double *saida, const double *tX, const double *tY, const double *vX, const double *alpha, const double gama, const int index, const int width, const int max);

__global__ void trainingKernelLoop(double *sum, const double *alpha, const double *x, const double *y, const double gama, const int nFeatures, const int nSamples, const int batchStart, const int batchEnd);

__global__ void trainingKernelFinishMultiple(double *alpha, double *sum, const double *y, const int nSamples, double *step, double *last, const double C);

__global__ void trainingKernelFinishSingle(double *alpha, double *sum, const double *y, const int nSamples, double step, double *last, const double C);

__global__ void initArray(double *array, const double value, const int max);