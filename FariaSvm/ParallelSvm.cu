#include "stdafx.h"
#include "ParallelSvm.cuh"
#include "SequentialKernel.h"
#include "ParallelKernel.cuh"
#include "MemoKernel.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"

#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__,  __LINE__,cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); } } 

using namespace std;

__global__ void partialSum(double *saida, const double *x, const double *alphaY, const double gama, const int posI, const int nFeatures,const int max)
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

ParallelSvm::ParallelSvm(int argc, char** argv, DataSet *ds)
{
	Logger::Stats("SVM", "Parallel");
	Logger::FunctionStart("ParallelSvm");
	_ds = ds;

	int halfGigaByte = 1 << 29;
	long gpuByteSize = ds->nSamples*ds->nFeatures*sizeof(double);

	if (gpuByteSize > halfGigaByte || gpuByteSize<0)
		throw exception("gpuByteSize to big for gpu.");

	string arg = Utils::GetComandVariable(argc, argv, "-tpb");
	if (!Utils::TryParseInt(arg,_threadsPerBlock) ||_threadsPerBlock % 32 != 0)
		_threadsPerBlock = 128;
	_blocks = 1 + _ds->nSamples / _threadsPerBlock;

	Logger::Stats("Blocks", _blocks);
	Logger::Stats("ThreadsPerBlock", _threadsPerBlock);
	Logger::Stats("Threads", _threadsPerBlock * _blocks);


	double* hst_x = (double*)malloc(sizeof(double)*_ds->nSamples*_ds->nFeatures);
	hst_s = (double*)malloc(sizeof(double)*_ds->nSamples);
	hst_aY = (double*)malloc(sizeof(double)*_ds->nSamples);

	for (int i = 0; i < _ds->nSamples; i++)
		for (int j = 0; j < _ds->nFeatures; j++)
			hst_x[i*_ds->nFeatures + j] = _ds->X[i][j];

	g = 1 / (2 * _ds->Gama*_ds->Gama);

	CUDA_SAFE_CALL(cudaSetDevice(0));

	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x, _ds->nSamples * _ds->nFeatures * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_s, _ds->nSamples * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_aY, _ds->nSamples * sizeof(double)));


	CUDA_SAFE_CALL(cudaMemcpy(dev_x, hst_x, _ds->nSamples * _ds->nFeatures * sizeof(double), cudaMemcpyHostToDevice));

	free(hst_x);
	Logger::FunctionEnd();
}

ParallelSvm::~ParallelSvm()
{
	cudaFree(dev_x);
	cudaFree(dev_s);
	cudaFree(dev_aY);
	free(hst_s);
	free(hst_aY);
}

void ParallelSvm::CopyResultToGpu(vector<double>& alpha)
{
	for (int i = 0; i < _ds->nSamples; i++)
	{
		hst_aY[i] = alpha[i] * _ds->Y[i];
	}

	CUDA_SAFE_CALL(cudaMemcpy(dev_aY, hst_aY, _ds->nSamples * sizeof(double), cudaMemcpyHostToDevice));
}

int ParallelSvm::Classify(int index, vector<double>& alpha, double& b)
{
	partialSum << <_blocks, _threadsPerBlock >> >(dev_s, dev_x, dev_aY, g, index, _ds->nFeatures, _ds->nSamples);

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
	alpha.clear();
	vector<double> oldAlpha;
	int samples = _ds->nSamples;
	for (int i = 0; i < samples; ++i){
		alpha.push_back(0);
		oldAlpha.push_back(1);
	}
	vector<vector<double>> x = _ds->X;
	vector<double> y = _ds->Y;
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = _ds->Step;
	double C = _ds->C;
	double precision = _ds->Precision;
	do
	{
		count++;

		difAlpha = 0;
		for (int i = 0; i < samples; ++i){
			difAlpha += alpha[i] - oldAlpha[i];
			oldAlpha[i] = alpha[i];
		}

		if (count>1)
			Logger::ClassifyProgress(count, step, lastDif, difAlpha);

		if (abs(difAlpha) < precision)
			break;
		if (abs(difAlpha - lastDif) > difAlpha / 10.0)
			step = step / 2;
		lastDif = difAlpha;
		if (count>1)
			CopyResultToGpu(oldAlpha);
		for (int i = 0; i < samples; ++i)
		{
			if (i == validationStart){
				i = validationEnd;
				if (i == samples)break;
			}

			double cudaSum = 0.0;
			if (count > 1){
				partialSum << <_blocks, _threadsPerBlock >> >(dev_s, dev_x, dev_aY, g, i, _ds->nFeatures, _ds->nSamples);

				CUDA_SAFE_CALL(cudaGetLastError());

				CUDA_SAFE_CALL(cudaDeviceSynchronize());

				CUDA_SAFE_CALL(cudaMemcpy(hst_s, dev_s, _ds->nSamples * sizeof(double), cudaMemcpyDeviceToHost));

				for (int i = 0; i < _ds->nSamples; i++)
					cudaSum += hst_s[i];
			}
			double value = oldAlpha[i] + step - step*y[i] * cudaSum;
			if (value > C)
				alpha[i] = C;
			else if (value < 0)
				alpha[i] = 0.0;
			else
				alpha[i] = value;
		}

	} while (true);
	int nSupportVectors = 0;
	for (int i = validationStart; i < validationEnd; i++)
		alpha[i] = 0;
	for (int i = 0; i < samples; ++i){
		if (i == validationStart){
			i = validationEnd;
			if (i == samples)break;
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
