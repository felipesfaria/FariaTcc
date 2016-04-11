#include "stdafx.h"
#include "ParallelSvm.cuh"
#include "Logger.h"
#include <cuda_runtime_api.h>
#include "Settings.h"
#include "CudaKernels.cuh"

#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__,  __LINE__,cudaGetErrorString(err)); \
      throw exception(cudaGetErrorString(err)); } } 

using namespace std;

CudaArray::~CudaArray()
{
	if (device!=NULL)
		cudaFree(device);
	if (deviceOnly && host != NULL)
		free(host);
}

void CudaArray::Init(int size)
{
	deviceOnly = true;
	if (size != this->size)
	{
		if (host!=NULL)
			free(host);
		host = (double*)malloc(size*sizeof(double));
	}
	Init(host, size);
}

void CudaArray::SetAll(int value)
{
	deviceOnly = true;
	if (size != this->size)
	{
		if (host != NULL)
			free(host);
		host = (double*)malloc(size*sizeof(double));
	}
	Init(host, size);
}

void CudaArray::Init(double* host, int size)
{
	if (size != this->size)
	{
		if (device!=NULL)
			cudaFree(device);
		CUDA_SAFE_CALL(cudaMalloc((void**)&device, size * sizeof(double)));
	}
	this->size = size;
	this->host = host;
}

void CudaArray::CopyToDevice()
{
	CUDA_SAFE_CALL(cudaMemcpy(device, host, size* sizeof(double), cudaMemcpyHostToDevice));
}

void CudaArray::CopyToHost()
{
	CUDA_SAFE_CALL(cudaMemcpy(host, device, size* sizeof(double), cudaMemcpyDeviceToHost));
}

double CudaArray::GetSum()
{
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += host[i];
	}
	return sum;
}

ParallelSvm::ParallelSvm(DataSet *ds)
	: BaseSvm(ds)
{
	int halfGigaByte = 1 << 29;
	long gpuByteSize = ds->nSamples*ds->nFeatures*sizeof(double);

	if (gpuByteSize > halfGigaByte || gpuByteSize < 0)
		throw exception("gpuByteSize to big for gpu.");

	Settings::instance()->GetUnsigned("threadsPerBlock", _threadsPerBlock);

	CUDA_SAFE_CALL(cudaSetDevice(0));
}

ParallelSvm::~ParallelSvm()
{
}

int ParallelSvm::Classify(TrainingSet *ts, int index)
{
	auto m = Logger::instance()->StartMetric("Classify");
	classificationKernel << <_blocks, _threadsPerBlock >> >(caSum.device, caTrainingX.device, caTrainingY.device, caValidationX.device, caAlpha.device, g, index, ts->width, ts->height);

	caSum.CopyToHost();

	double cudaSum = caSum.GetSum();

	auto precision = 0;
	auto sign = cudaSum - ts->b;
	m->Stop();
	if (sign > precision)
		return 1;
	if (sign < -precision)
		return -1;
	return 0;
}

void ParallelSvm::UpdateBlocks(TrainingSet *ts)
{
	int newBlockSize = (ts->height + _threadsPerBlock - 1) / _threadsPerBlock;;
	if (newBlockSize == _blocks) return;
	if (newBlockSize<1)
	{
		_blocks = 1;
		return;
	}
	_blocks = newBlockSize;
	Logger::instance()->Stats("Blocks", _blocks);
	Logger::instance()->Stats("Threads", _threadsPerBlock * _blocks);
}

void ParallelSvm::Train(TrainingSet *ts)
{
	auto m = Logger::instance()->StartMetric("Train");
	UpdateBlocks(ts);
	caTrainingX.Init(ts->x, ts->height*ts->width);
	caTrainingX.CopyToDevice();
	caTrainingY.Init(ts->y, ts->height);
	caTrainingY.CopyToDevice();

	caSum.Init(ts->height);

	caStep.Init(ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caStep.device, _initialStep);

	caLastDif.Init(ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caLastDif.device, 0.0);

	caAlpha.Init(ts->alpha, ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caAlpha.device, _initialStep);

	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	int batchSize = 512;
	int batchStart;
	int batchEnd;
	double avgStep;
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	do
	{
		for (batchStart = 0; batchStart < ts->height; batchStart += batchSize)
		{
			if (batchStart + batchSize>ts->height)
				batchEnd = ts->height;
			else
				batchEnd = batchStart + batchSize;
			trainingKernelLoop << <_blocks, _threadsPerBlock >> >(caSum.device, caAlpha.device, caTrainingX.device, caTrainingY.device, g, ts->width, ts->height, batchStart, batchEnd);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}
		trainingKernelFinish << <_blocks, _threadsPerBlock >> >(caAlpha.device, caSum.device, caTrainingY.device, ts->height, caStep.device, caLastDif.device, C);
		caLastDif.CopyToHost();
		caStep.CopyToHost();
#ifdef _DEBUG
		caAlpha.CopyToHost();
#endif
		difAlpha = caLastDif.GetSum() / ts->height;
		avgStep = caStep.GetSum() / ts->height;

		count++;

		Logger::instance()->TrainingProgress(count, avgStep, lastDif, difAlpha);

	} while ((abs(difAlpha) > Precision && count < MaxIterations));

	Logger::instance()->AddIntMetric("Iterations", count);
	caAlpha.CopyToHost();
	m->Stop();
}

void ParallelSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	auto m = Logger::instance()->StartMetric("Test");
	caValidationX.Init(vs->x, vs->height*vs->width);
	caValidationX.CopyToDevice();
	caSum.Init(ts->height);
	for (auto i = 0; i < vs->height; ++i)
	{
		int classifiedY = Classify(ts,i);
		vs->Validate(i, classifiedY);
	}
	m->Stop();
}
