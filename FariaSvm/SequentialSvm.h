#pragma once
#include "BaseSvm.h"

class SequentialSvm:
	public BaseSvm
{
public:
	double g;
	SequentialSvm(int argc, char** argv, DataSet *ds);
	~SequentialSvm();
	int Classify(int index, vector<double>& alpha, double& b) override;
	void Train(int validationStart, int validationEnd, vector<double>& alpha, double& b) override;
	void Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect) override;
private:
	double K(vector<double> x, vector<double> y);
};

