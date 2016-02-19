#include "stdafx.h"
#include "MemoKernel.h"
#include "DataSet.h"


MemoKernel::MemoKernel()
{
}


MemoKernel::~MemoKernel()
{
	free(_memo);
}

void MemoKernel::Init(DataSet ds)
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
	samples = ds.nSamples;
	auto size = ds.nSamples*ds.nSamples;
	_memo = (double*)malloc(size*sizeof(double));
	x = (double*)malloc(features*samples*sizeof(double));
	int i = 0;
	for (int i = 0; i < samples; i++)
		for (int j = 0; j < features; j++)
			x[i*features + j] = ds.X[i][j];
	for (int i = 0; i < samples; i++){
		for (int j = 0; j < samples; j++){
			_memo[i*samples + j] = Gauss(i, j);
		}
	}
}

double MemoKernel::Gauss(int i, int j)
{
	double sum = 0;
	double product;
	int rowI = i*features;
	int rowJ = j*features;
	for (int k = 0; k < features; ++k)
	{
		product = x[rowI + k] - x[rowJ + k];
		product *= product;
		sum += product;
	}
	return exp(-_sigma*sum);
}

double MemoKernel::K(int i, int j)
{
	int index = i*samples + j;
	return _memo[index];
}