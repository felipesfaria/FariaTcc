#include "stdafx.h"
#include "MemoKernel.h"
#include "DataSet.h"
#include "Logger.h"


MemoKernel::MemoKernel(const DataSet& ds)
{
	Logger::Stats("Kernel:", "Memo");
	_type = ds.kernelType;
	switch (_type)
	{
	case GAUSSIAN:
		_sigma = 1 / (2 * ds.Gama*ds.Gama);
		break;
	default:
		throw(new std::exception("Not Implemented exception"));
	}
	samples = ds.nSamples;
	features = ds.nFeatures;

	Logger::Stats("MemoByteSize:", GetMemoByteSize());

	auto size = ds.nSamples*ds.nSamples;
	_memo = (double*)malloc(size*sizeof(double));
	x = (double*)malloc(features*samples*sizeof(double));
	int i = 0;
	for (int i = 0; i < samples; i++)
		for (int j = 0; j < features; j++)
			x[i*features + j] = ds.X[i][j];

	for (int i = 0; i < samples; i++){
		for (int j = 0; j < samples; j++)
			_memo[i*samples + j] = Gauss(i, j);
	}
}


MemoKernel::~MemoKernel()
{
	free(_memo);
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

double MemoKernel::K(int i, int j, const DataSet& ds)
{
	int index = i*samples + j;
	return _memo[index];
}

int MemoKernel::GetMemoByteSize()
{
	return samples*samples*sizeof(double);
}
