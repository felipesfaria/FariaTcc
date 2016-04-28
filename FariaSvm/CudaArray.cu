#include "stdafx.h"
#include "CudaArray.cuh"
#include <memory>

using namespace std;
using namespace FariaSvm;

CudaArray::CudaArray(Settings settings)
{
	settings.GetUnsigned("threadsPerBlock", _threadsPerBlock);
}

CudaArray::~CudaArray()
{
	if (device != nullptr)
		cudaFree(device);
	if (deviceOnly && host != nullptr)
		delete[] host;
}

void CudaArray::Init(int size)
{
	deviceOnly = true;
	if (size != this->size)
	{
		if (host != nullptr)
			delete[] host;
		host = new double[size];
	}
	Init(host, size);
}

void CudaArray::SetAll(double value) const
{
	auto blocks = (size + _threadsPerBlock - 1) / _threadsPerBlock;
	initArray << <blocks, _threadsPerBlock >> >(device, value, size);
}

void CudaArray::Init(double* host, int size)
{
	if (size != this->size)
	{
		if (device != nullptr)
			cudaFree(device);
		CUDA_SAFE_CALL(cudaMalloc((void**)&device, size * sizeof(double)));
	}
	this->size = size;
	this->host = host;
}

void CudaArray::CopyToDevice() const
{
	CUDA_SAFE_CALL(cudaMemcpy(device, host, size* sizeof(double), cudaMemcpyHostToDevice));
}

void CudaArray::CopyToHost() const
{
	CUDA_SAFE_CALL(cudaMemcpy(host, device, size* sizeof(double), cudaMemcpyDeviceToHost));
}

double CudaArray::GetSum() const
{
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += host[i];
	return sum;
}