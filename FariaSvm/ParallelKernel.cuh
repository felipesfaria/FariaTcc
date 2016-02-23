#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BaseKernel.h"

class ParallelKernel : public BaseKernel
{
public:
	ParallelKernel(const DataSet& ds);
	~ParallelKernel();
	virtual double K(int i, int j, const DataSet& ds);
private:
	cudaError_t AddWithCuda(double *c, const double *a, const double *b, unsigned int size);
	double *dev_x = 0;
	double *dev_s = 0;
	double *hst_s;
	int *dev_i = 0;
	int *hst_i;
	int *dev_j = 0;
	int *hst_j;
	int features;
};

