#include "stdafx.h"
#include "SequentialSvm.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"
using namespace std;


SequentialSvm::SequentialSvm(int argc, char** argv, DataSet *ds)
	: BaseSvm(argc, argv, ds)
{
	Logger::Stats("SVM", "Sequential");
}

SequentialSvm::~SequentialSvm()
{
}

int SequentialSvm::Classify(int index, vector<double>& alpha, double& b)
{
	auto y = _ds->Y;
	auto precision = 0;
	auto size = alpha.size();
	auto sum = 0.0;
	for (auto i = 0; i < alpha.size(); ++i)
	{
		if (alpha[i] == 0) continue;
		sum += alpha[i] * y[i] * K(_ds->X[i],_ds->X[index]);
	}
	auto sign = sum - b;
	if (sign > precision)
		return 1;
	if (sign < -precision)
		return -1;
	return 0;
}

void SequentialSvm::Train(int validationStart, int validationEnd, vector<double>& alpha, double& b)
{
	Logger::FunctionStart("Train");
	alpha.clear();
	vector<double> oldAlpha;
	int samples = _ds->nSamples;
	for (int i = 0; i < samples; ++i){
		alpha.push_back(0);
		oldAlpha.push_back(1);
	}
	b = 0.0;
	vector<double> y = _ds->Y;
	int count = 0;
	double lastDif = 0.0;
	double difAlpha;
	double step = Step;
	double C = _ds->C;
	do
	{
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
				sum += y[j] * oldAlpha[j] * K(_ds->X[j],_ds->X[i]);
			}
			double value = oldAlpha[i] + step - step*y[i] * sum;
			if (value > C)
				alpha[i] = C;
			else if (value < 0)
				alpha[i] = 0.0;
			else
				alpha[i] = value;
		}

		difAlpha = 0;
		for (int i = 0; i < _ds->nSamples; ++i){
			difAlpha += alpha[i] - oldAlpha[i];
			oldAlpha[i] = alpha[i];
		}

		Logger::ClassifyProgress(count, step, lastDif, difAlpha);

		if (abs(difAlpha - lastDif) > difAlpha / 10.0)
			step = step / 2;
		lastDif = difAlpha;
		count++;
	} while ((abs(difAlpha) > Precision && count < 100) || count <= 1);
	int nSupportVectors = 0;
	for (int i = 0; i < samples; ++i){
		if (i == validationStart){
			i = validationEnd;
			if (i == samples)break;
		}
		if (alpha[i] > 0)
			nSupportVectors++;
	}
	Logger::Stats("nSupportVectors", nSupportVectors);
	Logger::FunctionEnd();
}

void SequentialSvm::Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect)
{
	Logger::FunctionStart("Test");
	auto start = clock();
	nCorrect = 0;
	int nSamples = _ds->nSamples;
	int nValidators = validationEnd - validationStart;
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

double SequentialSvm::K(vector<double> x, vector<double> y)
{
	double sum = 0;
	double product;
	for (int i = 0; i < x.size(); ++i)
	{
		product = x[i] - y[i];
		product *= product;
		sum += product;
	}
	return exp(-g*sum);
}