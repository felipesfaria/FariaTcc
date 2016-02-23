#include "stdafx.h"
#include "BaseKernel.h"
#include "Enums.h"

BaseKernel::BaseKernel()
{
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

double BaseKernel::K(int i, int j, const DataSet& ds)
{
	throw(new std::exception("Not Implemented exception"));
}