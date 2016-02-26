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
      exit(EXIT_FAILURE); } } 

using namespace std;

__global__ void classificationKernel(double *saida, const double *x, const double *alphaY, const double gama, const int posI, const int nFeatures, const int max)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//Sem isso ocorre erro com alguns conjuntos de dados
	if (idx > max) return;
	int aIndex = posI * nFeatures;
	int bIndex = idx*nFeatures;
	double sum = 0;
	double product;
	for (int i = 0; i < nFeatures; ++i)
	{
		product = x[aIndex + i] - x[bIndex + i];
		product *= product;
		sum += product;
	}
	saida[idx] = alphaY[idx] * exp(-gama * sum);
}

__global__ void trainingKernel(double *alpha, const double *x, const double *y, const double gama, const int nFeatures, const int nSamples, const double step, const double C, const int validationStart, const int validationEnd)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//Sem isso ocorre erro com alguns conjuntos de dados
	if (idx > nSamples || (idx >= validationStart && idx < validationEnd)) return;
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

ParallelSvm::ParallelSvm(int argc, char** argv, DataSet *ds)
	: BaseSvm(argc,argv,ds)
{
	Logger::FunctionStart("ParallelSvm");

	int halfGigaByte = 1 << 29;
	long gpuByteSize = ds->nSamples*ds->nFeatures*sizeof(double);

	if (gpuByteSize > halfGigaByte || gpuByteSize < 0)
		throw exception("gpuByteSize to big for gpu.");

	string arg = Utils::GetComandVariable(argc, argv, "-tpb");
	if (!Utils::TryParseInt(arg, _threadsPerBlock) || _threadsPerBlock % 32 != 0)
		_threadsPerBlock = 128;
	_blocks = 1 + _ds->nSamples / _threadsPerBlock;

	Logger::Stats("Blocks", _blocks);
	Logger::Stats("ThreadsPerBlock", _threadsPerBlock);
	Logger::Stats("Threads", _threadsPerBlock * _blocks);


	double* hst_x = (double*)malloc(sizeof(double)*_ds->nSamples*_ds->nFeatures);
	double* hst_y = (double*)malloc(sizeof(double)*_ds->nSamples);
	hst_s = (double*)malloc(sizeof(double)*_ds->nSamples);
	hst_a = (double*)malloc(sizeof(double)*_ds->nSamples);

	for (int i = 0; i < _ds->nSamples; i++)
		for (int j = 0; j < _ds->nFeatures; j++)
			hst_x[i*_ds->nFeatures + j] = _ds->X[i][j];

	for (int i = 0; i < _ds->nSamples; i++)
		hst_y[i] = _ds->Y[i];

	g = 1 / (2 * _ds->Gama*_ds->Gama);

	CUDA_SAFE_CALL(cudaSetDevice(0));

	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x, _ds->nSamples * _ds->nFeatures * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_s, _ds->nSamples * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_a, _ds->nSamples * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_y, _ds->nSamples * sizeof(double)));


	CUDA_SAFE_CALL(cudaMemcpy(dev_x, hst_x, _ds->nSamples * _ds->nFeatures * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_y, hst_y, _ds->nSamples * sizeof(double), cudaMemcpyHostToDevice));

	free(hst_x);
	free(hst_y);
	Logger::FunctionEnd();
}

ParallelSvm::~ParallelSvm()
{
	cudaFree(dev_x);
	cudaFree(dev_s);
	cudaFree(dev_a);
	cudaFree(dev_y);
	free(hst_s);
	free(hst_a);
}

void ParallelSvm::CopyResultToGpu(vector<double>& alpha)
{
	for (int i = 0; i < _ds->nSamples; i++)
	{
		hst_a[i] = alpha[i] * _ds->Y[i];
	}

	CUDA_SAFE_CALL(cudaMemcpy(dev_a, hst_a, _ds->nSamples * sizeof(double), cudaMemcpyHostToDevice));
}

int ParallelSvm::Classify(int index, vector<double>& alpha, double& b)
{
	classificationKernel << <_blocks, _threadsPerBlock >> >(dev_s, dev_x, dev_a, g, index, _ds->nFeatures, _ds->nSamples);

	CUDA_SAFE_CALL(cudaGetLastError());

	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaMemcpy(hst_s, dev_s, _ds->nSamples * sizeof(double), cudaMemcpyDeviceToHost));

	double cudaSum = 0.0;
	for (int i = 0; i < _ds->nSamples; i++)
		cudaSum += hst_s[i];

	auto precision = 0;
	auto sign = cudaSum - b;
	if (sign > precision)
		return 1;
	if (sign < -precision)
		return -1;
	return 0;
}

void ParallelSvm::Train(int validationStart, int validationEnd, vector<double>& alpha, double& b)
{
	Logger::FunctionStart("Train");
	vector<double> oldAlpha(_ds->nSamples);
	for (int i = 0; i < _ds->nSamples; i++)
		oldAlpha[i] = 0;
	initArray << <_blocks, _threadsPerBlock >> >(dev_a, 0.0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = Step;
	double C = _ds->C;
	do
	{
		trainingKernel << <_blocks, _threadsPerBlock >> >(dev_a, dev_x, dev_y, g, _ds->nFeatures, _ds->nSamples, step, _ds->C, validationStart, validationEnd);

		CUDA_SAFE_CALL(cudaGetLastError());

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		CUDA_SAFE_CALL(cudaMemcpy(hst_a, dev_a, _ds->nSamples * sizeof(double), cudaMemcpyDeviceToHost));

		difAlpha = 0;
		for (int i = 0; i < _ds->nSamples; ++i){
			difAlpha += hst_a[i] - oldAlpha[i];
			oldAlpha[i] = hst_a[i];
		}

		Logger::ClassifyProgress(count, step, lastDif, difAlpha);

		if (abs(difAlpha - lastDif) > difAlpha / 10.0)
			step = step / 2;
		lastDif = difAlpha;
		count++;
	} while ((abs(difAlpha) > Precision && count < 100) || count <= 1);
	alpha.clear();
	for (int i = 0; i < _ds->nSamples; i++)
		alpha.push_back(hst_a[i]);
	int nSupportVectors = 0;
	for (int i = validationStart; i < validationEnd; i++)
		alpha[i] = 0;
	for (int i = 0; i < _ds->nSamples; ++i){
		if (i == validationStart){
			i = validationEnd;
			if (i == _ds->nSamples)break;
		}
		if (alpha[i] != 0)
			nSupportVectors++;
	}
	b = 0.0;
	Logger::Stats("nSupportVectors", nSupportVectors);
	Logger::FunctionEnd();
}

void ParallelSvm::Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect)
{
	Logger::FunctionStart("Test");
	auto start = clock();
	nCorrect = 0;
	int nSamples = _ds->nSamples;
	int nValidators = validationEnd - validationStart;
	CopyResultToGpu(alpha1);
	for (auto i = validationStart; i < validationEnd; ++i)
	{
		int classifiedY = Classify(i, alpha1, b1);
		if (classifiedY == _ds->Y[i]){
			nCorrect++;
		}
	}
	Logger::Stats("AverageClassificationTime ", (clock() - start) / nValidators);
	auto percentageCorrect = static_cast<double>(nCorrect) / nValidators;
	Logger::Percentage(nCorrect, nValidators, percentageCorrect);
	Logger::FunctionEnd();
}
