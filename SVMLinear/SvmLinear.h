#pragma once
#include "DataSet.h"
#include "LinearKernel.h"

class SvmLinear
{
public:
	SvmLinear();
	~SvmLinear();
	int Classify(DataSet& ds, int index, vector<double>& alpha, LinearKernel& kernel, double& b);
	void Train(DataSet& ds, int nTrainers, LinearKernel& kernel, vector<double>& alpha, double& b);
	void Test(DataSet& ds, int nValidators, int nTrainers, LinearKernel& kernel, vector<double>& alpha1, double& b1, int& nCorrect);
private:
};

