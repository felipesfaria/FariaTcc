#pragma once
#include "SvmKernel.h"
class MemoKernel :
	public BaseKernel
{
public:
	MemoKernel();
	~MemoKernel();
	void Init(DataSet ds) override;
	double Gauss(int i, int j);
	double K(int i, int j) override;
private:
	double *_memo;
	double *x;
	int samples;
	int features;
};

