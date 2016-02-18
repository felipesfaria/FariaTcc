#pragma once
#include "DataSet.h"
#include "LinearKernel.h"

class SvmLinear
{
public:
	SvmKernel *kernel;
	SvmLinear();
	~SvmLinear();
	int Classify(DataSet& ds, int index, vector<double>& alpha, double& b);
	void Train(DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b);
	void Test(DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect);
private:
};

