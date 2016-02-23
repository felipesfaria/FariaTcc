#pragma once
#include "DataSet.h"
class BaseSvm
{
public:
	BaseSvm();
	virtual int Classify(int index, vector<double>& alpha, double& b);
	virtual void Train(int validationStart, int validationEnd, vector<double>& alpha, double& b);
	virtual void Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect);
	virtual ~BaseSvm();
protected:
	DataSet* _ds;
private:
};

