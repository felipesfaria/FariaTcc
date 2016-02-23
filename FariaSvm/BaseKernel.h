#pragma once
#include "DataSet.h"
#include "Enums.h"

class BaseKernel
{
public:
	BaseKernel();
	BaseKernel(const DataSet& ds);
	virtual ~BaseKernel();
	virtual double K(int i, int k, const DataSet& ds);
protected:
	KernelType _type;
	int _d;
	double _c;
	double _sigma;
};

