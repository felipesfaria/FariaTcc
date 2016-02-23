#pragma once
#include "DataSet.h"
class BaseSvm
{
public:
	BaseSvm();
	virtual int Classify(const DataSet& ds, int index, vector<double>& alpha, double& b);
	virtual void Train(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b);
	virtual void Test(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect);
	virtual ~BaseSvm();
};

