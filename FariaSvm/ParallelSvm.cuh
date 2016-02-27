#pragma once
#include "BaseSvm.h"
#include <driver_types.h>

class CudaArray
{
public:
	double* device = NULL;
	double* host = NULL;
	int size = 0;
	bool deviceOnly = false;
	~CudaArray();
	void Init(double* host, int size);
	void Init(int size);
	void CopyToDevice();
	void CopyToHost();
	double GetSum();
};

class ParallelSvm :
	public BaseSvm
{
public:
	ParallelSvm(int argc, char** argv, DataSet* ds);
	~ParallelSvm();
	int Classify(TrainingSet *ts, int index);
	void UpdateBlocks(TrainingSet* ts);
	void Train(TrainingSet *ts) override;
	void Test(TrainingSet *ts, ValidationSet *vs) override;
private:
	cudaError_t cudaStatus;
	int _blocks;
	int _threadsPerBlock;

	CudaArray caTrainingX;
	CudaArray caTrainingY;
	CudaArray caValidationX;
	CudaArray caAlpha;
	CudaArray caSum;
};

