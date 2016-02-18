#pragma once
#include <vector>

class DataSet;

enum KernelType
{
	NONE,
	LINEAR,
	HOMOGENEOUS_POLYNOMIAL,
	NONHOMOGENEOUS_POLYNOMIAL,
	GAUSSIAN

};
class SvmKernel
{
public:
	SvmKernel();
	virtual ~SvmKernel();
	virtual void Init(DataSet ds);
	virtual double K(std::vector<double> x, std::vector<double> y);
	virtual double Gaussian(std::vector<double> x, std::vector<double> y);
protected:
	KernelType _type;
	int _d;
	double _c;
	double _sigma;
};
