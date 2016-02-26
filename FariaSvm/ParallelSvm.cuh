#pragma once
#include "BaseSvm.h"
#include <driver_types.h>

class ParallelSvm :
	public BaseSvm
{
public:
	ParallelSvm(int argc, char** argv, DataSet* ds);
	~ParallelSvm();
	void CopyResultToGpu(vector<double>& alpha);
	int Classify(int index, vector<double>& alpha, double& b) override;
	void Train(int validationStart, int validationEnd, vector<double>& alpha, double& b) override;
	void Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect) override;
private:
	cudaError_t cudaStatus;;
	int _blocks;
	int _threadsPerBlock;

	double* dev_x;
	double* dev_s;
	double* hst_s;
	double* dev_a;
	double* hst_a;
	double* dev_y;
	double g;
};

