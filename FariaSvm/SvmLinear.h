#pragma once
#include "DataSet.h"
#include "BaseKernel.h"

class SvmLinear
{
public:
	BaseKernel *kernel;
	SvmLinear(int argc, char** argv, const DataSet& ds);
	~SvmLinear();
	int Classify(const DataSet& ds, int index, vector<double>& alpha, double& b);
	void Train(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b);
	void Test(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect);
private:
};

