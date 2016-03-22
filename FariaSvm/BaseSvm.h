#pragma once
#include "DataSet.h"
#include "Utils.h"
#include "TrainingSet.h"
#include "ValidationSet.h"
class BaseSvm
{
public:
	BaseSvm();
	BaseSvm(int argc, char** argv, DataSet* ds);
	virtual void Train(TrainingSet *ts);
	virtual void Test(TrainingSet *ts, ValidationSet *vs);
	virtual ~BaseSvm();
protected:
	DataSet* _ds;
	double Precision;
	double g;
	double Step;
	int MaxIterations=256;
private:
};

