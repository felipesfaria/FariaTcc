#include "stdafx.h"
#include "LinearKernel.h"
#include <vector>
using namespace std;

LinearKernel::~LinearKernel()
{
}

double LinearKernel::K(std::vector<double> x, std::vector<double> y)
{
	switch (_type)
	{
	case LINEAR:
		return Linear(x, y);
	case HOMOGENEOUS_POLYNOMIAL:
		return HomogeneousPolynomial(x, y);
	case NONHOMOGENEOUS_POLYNOMIAL:
		return NonHomogeneousPolynomial(x, y);
	case GAUSSIAN:
		return Gaussian(x, y);
	default:
		throw new std::exception("Kernel Type not defined or invalid type defined.");
	}
}
double LinearKernel::Linear(vector<double> x, vector<double> y){
	if (x.size() != y.size())
		throw new exception("Incompatible sizes..");
	double sum = 0;
	for (int i = 0; i < x.size(); ++i)
		sum += x[i] * y[i];
	return sum;
}

bool LinearKernel::DefineHomogeneousPolynomial(int d)
{
	if (d < 2)
		throw new exception("Invalid argument d<2");
	if (_type != NONE)
		throw new exception("Can't redefine Kernel");
	_d = d;
	_type = HOMOGENEOUS_POLYNOMIAL;
	return true;
}
double LinearKernel::HomogeneousPolynomial(vector<double> x, vector<double> y)
{
	double linear = Linear(x, y);
	double product = 1;
	for (int i = 0; i < _d; ++i)
		product *= linear;
	return product;
}

bool LinearKernel::DefineNonHomogeneousPolynomial(int d, double c)
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
double LinearKernel::NonHomogeneousPolynomial(vector<double> x, vector<double> y)
{
	double linear = Linear(x, y);
	double product = 1;
	for (int i = 0; i < _d; ++i)
		product *= linear + _c;
	return product;
}
double LinearKernel::Gaussian(vector<double> x, vector<double> y)
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
KernelType LinearKernel::GetType()
{
	return _type;
}