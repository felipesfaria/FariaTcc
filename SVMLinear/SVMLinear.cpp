#include "stdafx.h"
#include "SvmLinear.h"
#include "DataSet.h"
#include "LinearKernel.h"
#include "Logger.h"
#include <locale>
using namespace std;

SvmLinear::SvmLinear()
{
}


SvmLinear::~SvmLinear()
{
}


int SvmLinear::Classify(DataSet& ds, int index, vector<double>& alpha, double& b)
{
	auto x = ds.X;
	auto y = ds.Y;
	auto precision = 0;
	auto size = alpha.size();
	auto sum = 0.0;
	for (auto i = 0; i < alpha.size(); ++i)
	{
		if (alpha[i] == 0) continue;
		sum += alpha[i] * y[i] * kernel->K(x[i], x[index]);
	}
	auto sign = sum - b;
	if (sign > precision)
		return 1;
	if (sign < -precision)
		return -1;
	return 0;
}

void SvmLinear::Train(DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b)
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
				sum += y[j] * oldAlpha[j] * kernel->K(x[j], x[i]);
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

void SvmLinear::Test(DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect)
{
	Logger::FunctionStart("Test");
	auto start = clock();
	nCorrect = 0;
	int nSamples = ds.nSamples;
	int nValidators = validationEnd - validationStart;
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
