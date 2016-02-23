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

ParallelSvm::ParallelSvm(int argc, char** argv, const DataSet& ds)
{
	throw(exception("Not Implemented"));
}

ParallelSvm::ParallelSvm(int argc, char** argv, DataSet *ds)
{
	Logger::Stats("SVM:", "Parallel");
	this->ds = ds;
	string arg = Utils::GetComandVariable(argc, argv, "-k");

	int oneGigaByte = 1 << 30;
	long memoByteSize = ds->nSamples*ds->nSamples*sizeof(double);
	int halfGigaByte = 1 << 29;
	long gpuByteSize = ds->nSamples*ds->nFeatures*sizeof(double);
	switch (arg[0])
	{
	case 'm':
	case 'M':
		kernel = new MemoKernel(*ds);
		CopyAllToGpu();
		break;

	case 's':
	case 'S':
		throw(exception("Not Implemented"));
		kernel = new SequentialKernel(*ds);
		break;

	default:
		if (memoByteSize < oneGigaByte && memoByteSize>0)
			kernel = new MemoKernel(*ds);
		else
			kernel = new SequentialKernel(*ds);

		if (gpuByteSize < halfGigaByte && gpuByteSize>0)
			CopyAllToGpu();
		else
			throw exception("Not Implemented");
		break;
	}
}

ParallelSvm::~ParallelSvm()
{
	cudaFree(dev_x);
	cudaFree(dev_s);
	cudaFree(dev_aY);
	free(hst_s);
	free(hst_aY);
	free(kernel);
}

void ParallelSvm::CopyAllToGpu()
{
	Logger::FunctionStart("CopyAllToGpu");

	_threadsPerBlock = 256;
	Logger::Stats("ThreadsPerBlock:", _threadsPerBlock);
	_blocks = 1 + ds->nSamples / _threadsPerBlock;
	Logger::Stats("Blocks:", _blocks);

	double* hst_x = (double*)malloc(sizeof(double)*ds->nSamples*ds->nFeatures);
	hst_s = (double*)malloc(sizeof(double)*ds->nSamples);
	hst_aY = (double*)malloc(sizeof(double)*ds->nSamples);

	for (int i = 0; i < ds->nSamples; i++)
		for (int j = 0; j < ds->nFeatures; j++)
			hst_x[i*ds->nFeatures + j] = ds->X[i][j];

	g = 1 / (2 * ds->Gama*ds->Gama);

	CUDA_SAFE_CALL(cudaSetDevice(0));

	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x, ds->nSamples * ds->nFeatures * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_s, ds->nSamples * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_aY, ds->nSamples * sizeof(double)));


	CUDA_SAFE_CALL(cudaMemcpy(dev_x, hst_x, ds->nSamples * ds->nFeatures * sizeof(double), cudaMemcpyHostToDevice));

	free(hst_x);
	Logger::FunctionEnd();
}

void ParallelSvm::CopyResultToGpu(vector<double>& alpha)
{
	for (int i = 0; i < ds->nSamples; i++)
	{
		hst_aY[i] = alpha[i] * ds->Y[i];
	}

	CUDA_SAFE_CALL(cudaMemcpy(dev_aY, hst_aY, ds->nSamples * sizeof(double), cudaMemcpyHostToDevice));
}

int ParallelSvm::Classify(const DataSet& dst, int index, vector<double>& alpha, double& b)
{
	partialSum << <_blocks, _threadsPerBlock >> >(dev_s, dev_x, dev_aY, g, index, ds->nFeatures, ds->nSamples);

	CUDA_SAFE_CALL(cudaGetLastError());
	
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	
	CUDA_SAFE_CALL(cudaMemcpy(hst_s, dev_s, this->ds->nSamples * sizeof(double), cudaMemcpyDeviceToHost));

	double cudaSum = 0.0;
	for (int i = 0; i < this->ds->nSamples; i++)
		cudaSum += hst_s[i];

	auto precision = 0;
	auto sign = cudaSum - b;
	if (sign > precision)
		return 1;
	if (sign < -precision)
		return -1;
	return 0;
}

void ParallelSvm::Train(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b)
{
	Logger::FunctionStart("Train");
	alpha.clear();
	vector<double> oldAlpha;
	int samples = ds.nSamples;
	for (int i = 0; i < samples; ++i){
		alpha.push_back(0);
		oldAlpha.push_back(1);
	}
	vector<vector<double>> x = ds.X;
	vector<double> y = ds.Y;
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = ds.Step;
	double C = ds.C;
	double precision = ds.Precision;
	do
	{
		count++;

		difAlpha = 0;
		for (int i = 0; i < samples; ++i){
			if (i == validationStart)
				i = validationEnd;
			if (i == samples)break;
			difAlpha += alpha[i] - oldAlpha[i];
			oldAlpha[i] = alpha[i];
		}

		if (count>0)
			Logger::ClassifyProgress(count, step, lastDif, difAlpha);

		if (abs(difAlpha) < precision)
			break;
		if (abs(difAlpha - lastDif) > difAlpha / 10.0)
			step = step / 2;
		lastDif = difAlpha;
		for (int i = 0; i < samples; ++i)
		{
			if (i == validationStart){
				i = validationEnd;
				if (i == samples)break;
			}
			double sum = 0;
			for (int j = 0; j < samples; ++j)
			{
				if (j == validationStart){
					j = validationEnd;
					if (j == samples)break;
				}
				if (oldAlpha[j] == 0) continue;
				sum += y[j] * oldAlpha[j] * kernel->K(j, i, ds);
			}
			double value = oldAlpha[i] + step - step*y[i] * sum;
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

void ParallelSvm::Test(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect)
{
	Logger::FunctionStart("Test");
	auto start = clock();
	nCorrect = 0;
	int nSamples = ds.nSamples;
	int nValidators = validationEnd - validationStart;
	CopyResultToGpu(alpha1);
	for (auto i = validationStart; i < validationEnd; ++i)
	{
		int classifiedY = Classify(ds, i, alpha1, b1);
		if (classifiedY == ds.Y[i]){
			nCorrect++;
		}
	}
	Logger::Stats("AverageClassificationTime ", (clock() - start) / nValidators);
	auto percentageCorrect = static_cast<double>(nCorrect) / nValidators;
	Logger::Percentage(nCorrect, nValidators, percentageCorrect);
	Logger::FunctionEnd();
}
