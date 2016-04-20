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
using namespace FariaSvm;

CudaArray::~CudaArray()
{
	if (device!= nullptr)
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

void CudaArray::Init(double* host, int size)
{
	if (size != this->size)
	{
		if (device!= nullptr)
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
	{
		sum += host[i];
	}
	return sum;
}

ParallelSvm::ParallelSvm(shared_ptr<DataSet> ds)
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

int ParallelSvm::Classify(const TrainingSet& ts, const ValidationSet& vs, const int vIndex)
{
	if (!isTestPrepared)
		PrepareTest(ts, vs);

	auto m = Logger::instance()->StartMetric("Classify");
	classificationKernel << <_blocks, _threadsPerBlock >> >(caSum.device, caTrainingX.device, caTrainingY.device, caValidationX.device, caAlpha.device, g, vIndex, ts.width, ts.height);

	caSum.CopyToHost();

	double cudaSum = caSum.GetSum();

	auto value = cudaSum - ts.b;
	Logger::instance()->StopMetric(m);
	return SignOf(value);
}

void ParallelSvm::UpdateBlocks(TrainingSet & ts)
{
	int newBlockSize = (ts.height + _threadsPerBlock - 1) / _threadsPerBlock;;
	if (newBlockSize == _blocks) return;
	if (newBlockSize<1)
	{
		_blocks = 1;
		return;
	}
	_blocks = newBlockSize;
	Logger::instance()->Stats("Blocks", to_string(_blocks));
	Logger::instance()->Stats("Threads", to_string(_threadsPerBlock * _blocks));
}

void ParallelSvm::Train(TrainingSet & ts)
{
	auto m = Logger::instance()->StartMetric("Train");
	isTestPrepared = false;
	UpdateBlocks(ts);
	caTrainingX.Init(ts.x, ts.height*ts.width);
	caTrainingX.CopyToDevice();
	caTrainingY.Init(ts.y, ts.height);
	caTrainingY.CopyToDevice();
	
	double step = _initialStep;
	caSum.Init(ts.height);
	if (isMultiStep){
		caStep.Init(ts.height);
		initArray << <_blocks, _threadsPerBlock >> >(caStep.device, _initialStep, ts.height);
	}

	caLastDif.Init(ts.height);
	initArray << <_blocks, _threadsPerBlock >> >(caLastDif.device, 0.0, ts.height);

	caAlpha.Init(ts.alpha, ts.height);
	initArray << <_blocks, _threadsPerBlock >> >(caAlpha.device, _initialStep, ts.height);

	int count = 0;
	double lastDif = 999999;
	double avgDif;
	int batchSize = 512;
	int batchStart;
	int batchEnd;
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	do
	{
		for (batchStart = 0; batchStart < ts.height; batchStart += batchSize)
		{
			if (batchStart + batchSize>ts.height)
				batchEnd = ts.height;
			else
				batchEnd = batchStart + batchSize;
			trainingKernelLoop << <_blocks, _threadsPerBlock >> >(caSum.device, caAlpha.device, caTrainingX.device, caTrainingY.device, g, ts.width, ts.height, batchStart, batchEnd);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}

		if (isMultiStep)
			trainingKernelFinishMultiple << <_blocks, _threadsPerBlock >> >(caAlpha.device, caSum.device, caTrainingY.device, ts.height, caStep.device, caLastDif.device, C);
		else
			trainingKernelFinishSingle << <_blocks, _threadsPerBlock >> >(caAlpha.device, caSum.device, caTrainingY.device, ts.height, step, caLastDif.device, C);
		
		caLastDif.CopyToHost();
		avgDif = caLastDif.GetSum() / ts.height;

		if (!isMultiStep)
		{
			updateStep(step, lastDif, avgDif);
		}
		else{
			caStep.CopyToHost();
			step = caStep.GetSum() / ts.height;
		}

#ifdef _DEBUG
		caAlpha.CopyToHost();
#endif
		count++;

		Logger::instance()->TrainingProgress(count, step, avgDif);

	} while (abs(avgDif) > Precision && count < MaxIterations);

	Logger::instance()->AddIntMetric("Iterations", count);
	caAlpha.CopyToHost();
	Logger::instance()->StopMetric(m);
}

void ParallelSvm::PrepareTest(const TrainingSet& ts, const ValidationSet& vs)
{
	caValidationX.Init(vs.x, vs.height*vs.width);
	caValidationX.CopyToDevice();
	caSum.Init(ts.height);
	isTestPrepared = true;
}
