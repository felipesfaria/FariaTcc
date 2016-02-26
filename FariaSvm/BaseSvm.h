#pragma once
#include "DataSet.h"
#include "Utils.h"
class BaseSvm
{
public:
	BaseSvm();
	BaseSvm(int argc, char** argv, DataSet* ds);
	virtual int Classify(int index, vector<double>& alpha, double& b);
	virtual void Train(int validationStart, int validationEnd, vector<double>& alpha, double& b);
	virtual void Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect);
	virtual ~BaseSvm();
protected:
	DataSet* _ds;
	double Precision;
	double g;
	double Step;
private:
};

