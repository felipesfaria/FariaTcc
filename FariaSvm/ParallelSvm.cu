#include "stdafx.h"
#include "ParallelSvm.cuh"
#include "SequentialKernel.h"
#include "ParallelKernel.cuh"
#include "MemoKernel.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"
using namespace std;

__global__ void partialSum(double *saida, const double *x, const double *alphaY, const double *gama, const int *posI, const int *nFeatures)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int aIndex = posI[0] * nFeatures[0];
	int bIndex = idx*nFeatures[0];
	double sum = 0;
	double product;
	for (int i = 0; i < nFeatures[0]; ++i)
	{
		product = x[aIndex + i] - x[bIndex + i];
		product *= product;
		sum += product;
	}
	saida[idx] = alphaY[idx] * exp(-gama[0] * sum);
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
	int doubleSize = sizeof(double);
	long memoByteSize = (long)ds->nSamples*(long)ds->nSamples*(long)doubleSize;
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
		int oneGigaByte = 1 << 29;
		if (memoByteSize < oneGigaByte && memoByteSize>0){
			kernel = new MemoKernel(*ds);
			CopyAllToGpu();
		}
		else{
			throw(exception("Not Implemented"));
			kernel = new SequentialKernel(*ds);
		}
		break;
	}
}

ParallelSvm::~ParallelSvm()
{
	free(kernel);
}

void ParallelSvm::CopyAllToGpu()
{
	Logger::FunctionStart("CopyAllToGpu");

	double* hst_x = (double*)malloc(sizeof(double)*ds->nSamples*ds->nFeatures);
	hst_s = (double*)malloc(sizeof(double)*ds->nSamples);
	hst_aY = (double*)malloc(sizeof(double)*ds->nSamples);
	hst_g = (double*)malloc(sizeof(double));
	hst_i = (int*)malloc(sizeof(int));
	hst_f = (int*)malloc(sizeof(int));

	for (int i = 0; i < ds->nSamples; i++)
		for (int j = 0; j < ds->nFeatures; j++)
			hst_x[i*ds->nFeatures + j] = ds->X[i][j];

	hst_g[0] = 1 / (2 * ds->Gama*ds->Gama);
	hst_f[0] = ds->nFeatures;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		throw(exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"));
	}

	//
	cudaStatus = cudaMalloc((void**)&dev_x, ds->nSamples * ds->nFeatures * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_s, ds->nSamples * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_aY, ds->nSamples * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_g, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_i, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(exception("cudaMalloc failed!"));
	}
	cudaStatus = cudaMalloc((void**)&dev_f, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw(exception("cudaMalloc failed!"));
	}

	//Copy to GPU x
	cudaStatus = cudaMemcpy(dev_x, hst_x, ds->nSamples * ds->nFeatures * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(exception("cudaMemcpy failed!"));
	}
	cudaStatus = cudaMemcpy(dev_f, hst_f, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(exception("cudaMemcpy failed!"));
	}

	cudaStatus = cudaMemcpy(dev_g, hst_g, sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(exception("cudaMemcpy failed!"));
	}
	free(hst_x);
	Logger::FunctionEnd();
}

void ParallelSvm::CopyResultToGpu(vector<double>& alpha)
{
	for (int i = 0; i < ds->nSamples; i++)
	{
		hst_aY[i] = alpha[i] * ds->Y[i];
	}

	cudaStatus = cudaMemcpy(dev_aY, hst_aY, ds->nSamples * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw(exception("cudaMemcpy failed!"));
	}
}

int ParallelSvm::Classify(const DataSet& ds, int index, vector<double>& alpha, double& b)
{

	hst_i[0] = index;
	cudaMemcpy(dev_i, hst_i, sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 128;
	int blocks = 1 + ds.nSamples / threadsPerBlock;
	partialSum << <blocks, 128 >> >(dev_s, dev_x, dev_aY, dev_g, dev_i, dev_f);

	cudaMemcpy(hst_s, dev_s, ds.nSamples * sizeof(double), cudaMemcpyDeviceToHost);

	double cudaSum = 0.0;
	for (int i = 0; i < ds.nSamples; i++)
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
