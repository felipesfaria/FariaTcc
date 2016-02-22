#include "stdafx.h"
#include "BaseKernel.h"
#include "DataSet.h"
#include "Enums.h"

BaseKernel::BaseKernel()
{
	_type = NONE;
}

BaseKernel::~BaseKernel()
{
}

void BaseKernel::Init(DataSet ds)
{
	_type = ds.kernelType;
	switch (_type)
	{
	case GAUSSIAN:
		_sigma = ds.Gama;
		break;
	default:
		throw(new std::exception("Not Implemented exception"));
	}
}

double BaseKernel::K(std::vector<double> x, std::vector<double> y)
{
	throw(new std::exception("Not Implemented exception"));
}

double BaseKernel::K(int i, int k)
{
	throw(new std::exception("Not Implemented exception"));
}

double BaseKernel::Gaussian(std::vector<double> x, std::vector<double> y)
{
	throw(new std::exception("Not Implemented exception"));
}