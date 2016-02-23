#pragma once
#include "BaseSvm.h"
#include "BaseKernel.h"

class SequentialSvm:
	public BaseSvm
{
public:
	BaseKernel *kernel;
	SequentialSvm(int argc, char** argv, const DataSet& ds);
	~SequentialSvm();
	int Classify(const DataSet& ds, int index, vector<double>& alpha, double& b) override;
	void Train(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b) override;
	void Test(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect) override;
private:
};

