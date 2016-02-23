#pragma once
#include "BaseKernel.h"

class MemoKernel :
	public BaseKernel
{
public:
	void LoadMemo(const DataSet& ds);
	MemoKernel(const DataSet& ds);
	~MemoKernel();
	double Gauss(int i, int j);
	double K(int i, int j, const DataSet& ds) override;
	long GetMemoByteSize();
private:
	double *_memo;
	double *x;
	int samples;
	int features;
};

