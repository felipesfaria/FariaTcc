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
	virtual double K(std::vector<double> x, std::vector<double> y);
	virtual double K(int i, int k);
	virtual double Gaussian(std::vector<double> x, std::vector<double> y);
protected:
	KernelType _type;
	int _d;
	double _c;
	double _sigma;
};

