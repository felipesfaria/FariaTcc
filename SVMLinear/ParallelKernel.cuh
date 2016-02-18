#pragma once
#include "SvmKernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ParallelKernel : public SvmKernel
{
public:
	ParallelKernel();
	~ParallelKernel();
	void Init(DataSet ds) override;
	virtual double K(std::vector<double> x, std::vector<double> y);
	virtual double K(int i, int j);
private:
	cudaError_t AddWithCuda(double *c, const double *a, const double *b, unsigned int size);
};

