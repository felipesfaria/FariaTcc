#include "stdafx.h"
#include "SequentialKernel.h"
#include "DataSet.h"
#include "Logger.h"
using namespace std;

SequentialKernel::SequentialKernel(const DataSet& ds)
{
	Logger::Stats("Kernel", "Sequential");
	_type = ds.kernelType;
	switch (_type)
	{
	case GAUSSIAN:
		_sigma = ds.Gama;
		break;
	default:
		throw(exception("Not Implemented exception"));
	}
}
SequentialKernel::~SequentialKernel()
{
}

double SequentialKernel::K(int i, int j, const DataSet& ds)
{
	switch (_type)
	{
	case LINEAR:
		return Linear(ds.X[i], ds.X[j]);
	case HOMOGENEOUS_POLYNOMIAL:
		return HomogeneousPolynomial(ds.X[i], ds.X[j]);
	case NONHOMOGENEOUS_POLYNOMIAL:
		return NonHomogeneousPolynomial(ds.X[i], ds.X[j]);
	case GAUSSIAN:
		return Gaussian(ds.X[i], ds.X[j]);
	default:
		throw new std::exception("Kernel Type not defined or invalid type defined.");
	}
}
double SequentialKernel::Linear(vector<double> x, vector<double> y){
	if (x.size() != y.size())
		throw new exception("Incompatible sizes..");
	double sum = 0;
	for (int i = 0; i < x.size(); ++i)
		sum += x[i] * y[i];
	return sum;
}

bool SequentialKernel::DefineHomogeneousPolynomial(int d)
{
	if (d < 2)
		throw new exception("Invalid argument d<2");
	if (_type != NONE)
		throw new exception("Can't redefine Kernel");
	_d = d;
	_type = HOMOGENEOUS_POLYNOMIAL;
	return true;
}
double SequentialKernel::HomogeneousPolynomial(vector<double> x, vector<double> y)
{
	double linear = Linear(x, y);
	double product = 1;
	for (int i = 0; i < _d; ++i)
		product *= linear;
	return product;
}

bool SequentialKernel::DefineNonHomogeneousPolynomial(int d, double c)
{
	if (d < 2)
		throw new exception("Invalid argument d<2");
	if (c <= 0)
		throw new exception("Invalid argument c<=0");
	if (_type != NONE)
		throw new exception("Can't redefine Kernel");
	_d = d;
	_c = c;
	_type = NONHOMOGENEOUS_POLYNOMIAL;
	return true;
}
double SequentialKernel::NonHomogeneousPolynomial(vector<double> x, vector<double> y)
{
	double linear = Linear(x, y);
	double product = 1;
	for (int i = 0; i < _d; ++i)
		product *= linear + _c;
	return product;
}
double SequentialKernel::Gaussian(vector<double> x, vector<double> y)
{
	double sum = 0;
	double product;
	double gama = 1 / (2 * _sigma*_sigma);
	for (int i = 0; i < x.size(); ++i)
	{
		product = x[i] - y[i];
		product *= product;
		sum += product;
	}
	return exp(-gama*sum);
}
KernelType SequentialKernel::GetType()
{
	return _type;
}