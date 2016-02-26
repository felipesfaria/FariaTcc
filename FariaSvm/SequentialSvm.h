#pragma once
#include "BaseSvm.h"

class SequentialSvm:
	public BaseSvm
{
public:
	SequentialSvm(int argc, char** argv, DataSet *ds);
	~SequentialSvm();
	int Classify(TrainingSet* ts, double* sample);
	void Train(TrainingSet *ts) override;
	void Test(TrainingSet *ts, ValidationSet *vs) override;
	double K(double* x, double* y, int size);
private:
};

