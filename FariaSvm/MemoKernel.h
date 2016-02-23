#pragma once
#include "BaseKernel.h"

class MemoKernel :
	public BaseKernel
{
public:
	MemoKernel(const DataSet& ds);
	~MemoKernel();
	double Gauss(int i, int j);
	double K(int i, int j, const DataSet& ds) override;
	int GetMemoByteSize();
private:
	double *_memo;
	double *x;
	int samples;
	int features;
};

