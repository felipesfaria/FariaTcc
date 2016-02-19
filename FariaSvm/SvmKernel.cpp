#include "stdafx.h"
#include "SvmKernel.h"
#include "DataSet.h"


SvmKernel::SvmKernel()
{
	_type = NONE;
}

SvmKernel::~SvmKernel()
{
}

void SvmKernel::Init(DataSet ds)
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

double SvmKernel::K(std::vector<double> x, std::vector<double> y)
{
	throw(new std::exception("Not Implemented exception"));
}

double SvmKernel::K(int i, int k)
{
	throw(new std::exception("Not Implemented exception"));
}

double SvmKernel::Gaussian(std::vector<double> x, std::vector<double> y)
{
	throw(new std::exception("Not Implemented exception"));
}