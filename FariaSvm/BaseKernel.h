#pragma once
#include <vector>
#include "DataSet.h"
#include "Enums.h"

class BaseKernel
{
public:
	BaseKernel();
	virtual ~BaseKernel();
	virtual void Init(DataSet ds);
	virtual double K(int i, int k, const DataSet& ds);
protected:
	KernelType _type;
	int _d;
	double _c;
	double _sigma;
};

