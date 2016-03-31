#include "stdafx.h"
#include "ParallelSvm.cuh"
#include "Logger.h"
#include <locale>
#include "Utils.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__,  __LINE__,cudaGetErrorString(err)); \
      throw exception(cudaGetErrorString(err)); } } 

using namespace std;

__global__ void classificationKernel(double *saida, const double *tX, const double *tY, const double *vX, const double *alpha, const double gama, const int index, const int width, const int max)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > max)return;
	int vIndex = index * width;
	int tIndex = idx*width;
	double sum = 0;
	double product;
	for (int i = 0; i < width; ++i)
	{
		product = vX[vIndex + i] - tX[tIndex + i];
		product *= product;
		sum += product;
	}
	saida[idx] = tY[idx] * alpha[idx] * exp(-gama * sum);
}

__global__ void trainingKernelLoop(double *sum,const double *alpha, const double *x, const double *y, const double gama, const int nFeatures, const int nSamples,const int batchStart, const int batchEnd)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > nSamples) return;
	int bIndex = idx*nFeatures;
	double outerSum = sum[idx];
	for (int i = batchStart; i < batchEnd; i++){
		int aIndex = i * nFeatures;
		double product;
		double innerSum = 0;
		for (int j = 0; j < nFeatures; ++j)
		{
			product = x[aIndex + j] - x[bIndex + j];
			product *= product;
			innerSum += product;
		}
		outerSum += alpha[i] * y[i] * exp(-gama * innerSum);
	}
	sum[idx] = outerSum;
}

__global__ void trainingKernelFinish(double *alpha, const double *sum, const double *y, const int nSamples, double *step, double *last, const double C)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > nSamples) return;
	double deltaAlpha = step[idx] - step[idx]*y[idx] * sum[idx];
	double newAlpha = deltaAlpha + alpha[idx];
	if (newAlpha > C)
		newAlpha = C;
	else if (newAlpha < 0)
		newAlpha = 0.0;
	auto dif = newAlpha - alpha[idx];
	if (dif*last[idx] < 0)
		step[idx] /= 2;
	last[idx] = dif;
	alpha[idx] = newAlpha;
}

__global__ void initArray(double *array, const double value)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	array[idx] = value;
}

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

ParallelSvm::ParallelSvm(int argc, char** argv, DataSet *ds)
	: BaseSvm(argc, argv, ds)
{
	Logger::instance()->FunctionStart("ParallelSvm");

	int halfGigaByte = 1 << 29;
	long gpuByteSize = ds->nSamples*ds->nFeatures*sizeof(double);

	if (gpuByteSize > halfGigaByte || gpuByteSize < 0)
		throw exception("gpuByteSize to big for gpu.");

	string arg = Utils::GetComandVariable(argc, argv, "-tpb");
	if (!Utils::TryParseInt(arg, _threadsPerBlock) || _threadsPerBlock % 32 != 0)
		_threadsPerBlock = 512;

	CUDA_SAFE_CALL(cudaSetDevice(0));

	Logger::instance()->FunctionEnd("ParallelSvm");
}

ParallelSvm::~ParallelSvm()
{
}

int ParallelSvm::Classify(TrainingSet *ts, int index)
{
	classificationKernel << <_blocks, _threadsPerBlock >> >(caSum.device, caTrainingX.device, caTrainingY.device, caValidationX.device, caAlpha.device, g, index, ts->width, ts->height);

	caSum.CopyToHost();

	double cudaSum = caSum.GetSum();

	auto precision = 0;
	auto sign = cudaSum - ts->b;
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
	Logger::instance()->Stats("ThreadsPerBlock", _threadsPerBlock);
	Logger::instance()->Stats("Threads", _threadsPerBlock * _blocks);
}

void ParallelSvm::Train(TrainingSet *ts)
{
	Logger::instance()->FunctionStart("Train");
	UpdateBlocks(ts);
	caTrainingX.Init(ts->x, ts->height*ts->width);
	caTrainingX.CopyToDevice();
	caTrainingY.Init(ts->y, ts->height);
	caTrainingY.CopyToDevice();

	caSum.Init(ts->height);

	caStep.Init(ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caStep.device, Step);

	caLastDif.Init(ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caLastDif.device, 0.0);

	caAlpha.Init(ts->alpha, ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caAlpha.device, Step);

	double* oldAlpha = (double*)malloc(ts->height*sizeof(double));
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = Step;
	double C = _ds->C;
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
		trainingKernelFinish << <_blocks, _threadsPerBlock >> >(caAlpha.device, caSum.device, caTrainingY.device, ts->width, caStep.device, caLastDif.device, C);
		//caAlpha.CopyToHost();
		caLastDif.CopyToHost();
		caStep.CopyToHost();

		difAlpha = caLastDif.GetSum() / ts->height;
		avgStep = caStep.GetSum() / ts->height;

		Logger::instance()->ClassifyProgress(count, avgStep, lastDif, difAlpha);

		count++;
	} while ((abs(difAlpha) > Precision && count < MaxIterations) || count <=1);
	caAlpha.CopyToHost();
	int nSupportVectors = 0;
	for (int i = 0; i < ts->height; ++i){
		if (ts->alpha[i] != 0)
			nSupportVectors++;
	}
	Logger::instance()->Stats("nSupportVectors", nSupportVectors);
	Logger::instance()->FunctionEnd("Train");
}

void ParallelSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	Logger::instance()->FunctionStart("Test");
	auto start = clock();
	caValidationX.Init(vs->x, vs->height*vs->width);
	caValidationX.CopyToDevice();
	caSum.Init(ts->height);
	for (auto i = 0; i < vs->height; ++i)
	{
		int classifiedY = Classify(ts,i);
		if (classifiedY == vs->y[i]){
			vs->nCorrect++;
		}
		else if (classifiedY<0)
		{
			vs->nNegativeWrong++;
		}
		else if (classifiedY>0)
		{
			vs->nPositiveWrong++;
		}
		else
			vs->nNullWrong++;
	}
	Logger::instance()->Stats("nNegativeWrong ", vs->nNegativeWrong);
	Logger::instance()->Stats("nPositiveWrong ", vs->nPositiveWrong);
	Logger::instance()->Stats("nNullWrong ", vs->nNullWrong);
	Logger::instance()->Stats("AverageClassificationTime ", (clock() - start) / vs->height);
	auto percentageCorrect = static_cast<double>(vs->nCorrect) / vs->height;
	Logger::instance()->Percentage(vs->nCorrect, vs->height, percentageCorrect);
	Logger::instance()->FunctionEnd("Test");
}
