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

__global__ void trainingKernel(double *alpha, const double *x, const double *y, const double gama, const int nFeatures, const int nSamples, const double step, const double C)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx > nSamples) return;
	int bIndex = idx*nFeatures;
	double outerSum = 0;
	for (int i = 0; i < nSamples; i++){
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
	double deltaAlpha = step - step*y[idx] * outerSum;
	double newAlpha = deltaAlpha + alpha[idx];
	if (newAlpha > C)
		newAlpha = C;
	else if (newAlpha < 0)
		newAlpha = 0.0;
	alpha[idx] = newAlpha;
}

__global__ void initArray(double *array, const double value)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	array[idx] = value;
}
CudaArray::~CudaArray()
{
	if (initialized)
		cudaFree(device);
	if (deviceOnly)
		free(host);
}

void CudaArray::Init(int size)
{
	deviceOnly = true;
	if (size != this->size)
	{
		if (initialized)
			free(host);
		host = (double*)malloc(size*sizeof(double));
	}
	Init(host, size);
}

void CudaArray::Init(double* host, int size)
{
	if (size != this->size)
	{
		if (initialized)
			cudaFree(device);
		CUDA_SAFE_CALL(cudaMalloc((void**)&device, size * sizeof(double)));
	}
	initialized = true;
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
	Logger::FunctionStart("ParallelSvm");

	int halfGigaByte = 1 << 29;
	long gpuByteSize = ds->nSamples*ds->nFeatures*sizeof(double);

	if (gpuByteSize > halfGigaByte || gpuByteSize < 0)
		throw exception("gpuByteSize to big for gpu.");

	string arg = Utils::GetComandVariable(argc, argv, "-tpb");
	if (!Utils::TryParseInt(arg, _threadsPerBlock) || _threadsPerBlock % 32 != 0)
		_threadsPerBlock = 128;

	CUDA_SAFE_CALL(cudaSetDevice(0));

	Logger::FunctionEnd();
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
	Logger::Stats("Blocks", _blocks);
	Logger::Stats("ThreadsPerBlock", _threadsPerBlock);
	Logger::Stats("Threads", _threadsPerBlock * _blocks);
}

void ParallelSvm::Train(TrainingSet *ts)
{
	Logger::FunctionStart("Train");
	UpdateBlocks(ts);
	caTrainingX.Init(ts->x, ts->height*ts->width);
	caTrainingX.CopyToDevice();
	caTrainingY.Init(ts->y, ts->height);
	caTrainingY.CopyToDevice();

	caAlpha.Init(ts->alpha, ts->height);
	initArray << <_blocks, _threadsPerBlock >> >(caAlpha.device, Step);
	double* oldAlpha = (double*)malloc(ts->height*sizeof(double));
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = Step;
	double C = _ds->C;
	do
	{
		trainingKernel << <_blocks, _threadsPerBlock >> >(caAlpha.device, caTrainingX.device, caTrainingY.device, g, ts->width, ts->height, step, _ds->C);
		
		caAlpha.CopyToHost();

		difAlpha = 0;
		for (int i = 0; i < ts->height; ++i){
			difAlpha += ts->alpha[i] - oldAlpha[i];
			oldAlpha[i] = ts->alpha[i];
		}

		Logger::ClassifyProgress(count, step, lastDif, difAlpha);

		if (abs(difAlpha - lastDif) > difAlpha / 10.0)
			step = step / 2;
		lastDif = difAlpha;
		count++;
	} while ((abs(difAlpha) > Precision && count < 100) || count <= 1);

	int nSupportVectors = 0;
	for (int i = 0; i < ts->height; ++i){
		if (ts->alpha[i] != 0)
			nSupportVectors++;
	}
	Logger::Stats("nSupportVectors", nSupportVectors);
	Logger::FunctionEnd();
}

void ParallelSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	Logger::FunctionStart("Test");
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
	Logger::Stats("nNegativeWrong ", vs->nNegativeWrong);
	Logger::Stats("nPositiveWrong ", vs->nPositiveWrong);
	Logger::Stats("nNullWrong ", vs->nNullWrong);
	Logger::Stats("AverageClassificationTime ", (clock() - start) / vs->height);
	auto percentageCorrect = static_cast<double>(vs->nCorrect) / vs->height;
	Logger::Percentage(vs->nCorrect, vs->height, percentageCorrect);
	Logger::FunctionEnd();
}
