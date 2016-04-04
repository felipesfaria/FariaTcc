#pragma once
#include "BaseSvm.h"

class SequentialSvm:
	public BaseSvm
{
public:
	SequentialSvm(DataSet *ds);
	~SequentialSvm();
	int Classify(TrainingSet* ts, double* sample);
	void Train(TrainingSet *ts) override;
	void Test(TrainingSet *ts, ValidationSet *vs) override;
	double K(double* x, double* y, int size);
private:
};

