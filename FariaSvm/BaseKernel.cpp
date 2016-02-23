#include "stdafx.h"
#include "BaseKernel.h"
#include "Enums.h"

BaseKernel::BaseKernel()
{
}

BaseKernel::BaseKernel(const DataSet& ds)
{
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

BaseKernel::~BaseKernel()
{
}

double BaseKernel::K(int i, int j, const DataSet& ds)
{
	throw(exception("Not Implemented exception"));
}
