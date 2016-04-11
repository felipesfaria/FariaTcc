#pragma once
#include "DataSet.h"
#include "TrainingSet.h"
#include "ValidationSet.h"
class BaseSvm
{
public:
	BaseSvm();
	BaseSvm(DataSet* ds);
	virtual void Train(TrainingSet *ts);
	virtual void Test(TrainingSet *ts, ValidationSet *vs);
	virtual ~BaseSvm();
protected:
	DataSet* _ds;
	double Precision;
	double g;
	double C;
	double _initialStep;
	unsigned MaxIterations;
private:
};

