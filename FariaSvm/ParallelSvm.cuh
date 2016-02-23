#pragma once
#include "BaseSvm.h"
#include "BaseKernel.h"
#include <driver_types.h>

class ParallelSvm :
	public BaseSvm
{
public:
	BaseKernel *kernel;
	ParallelSvm(int argc, char** argv, const DataSet& ds);
	ParallelSvm(int argc, char** argv, DataSet* ds);
	~ParallelSvm();
	void CopyAllToGpu();
	void CopyResultToGpu(vector<double>& alpha);
	int Classify(const DataSet& ds, int index, vector<double>& alpha, double& b) override;
	void Train(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b) override;
	void Test(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect) override;
private:
	cudaError_t cudaStatus;;
	DataSet* ds;
	int _blocks;
	int _threadsPerBlock;

	double* dev_x;
	double* dev_s;
	double* hst_s;
	double* dev_aY;
	double* hst_aY;
	double g;
};

