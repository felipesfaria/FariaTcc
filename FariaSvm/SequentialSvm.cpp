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

int SequentialSvm::Classify(TrainingSet *ts, double* sample)
{
	auto sum = 0.0;
	for (auto i = 0; i < ts->height; ++i)
	{
		if (ts->alpha[i] == 0) continue;
		sum += ts->alpha[i] * ts->y[i] * K(ts->GetSample(i), sample,ts->width);
	}
	auto sign = sum - ts->b;
	if (sign > Precision)
		return 1;
	if (sign < Precision)
		return -1;
	return 0;
}

void SequentialSvm::Train(TrainingSet *ts)
{
	Logger::FunctionStart("Train");
	double* alpha = ts->alpha;
	vector<double> oldAlpha;
	int samples = _ds->nSamples;
	for (int i = 0; i < ts->height; ++i){
		alpha[i] = 0;
		oldAlpha.push_back(0);
	}
	int count = 0;
	double *lastDif = (double*)malloc(ts->height*sizeof(double));
	for (int i = 0; i < ts->height; i++)
		lastDif[i] = 0;
	double difAlpha;
	double *steps = (double*)malloc(ts->height*sizeof(double));
	for (int i = 0; i < ts->height; i++)
		steps[i] = Step;
	double step = Step;
	double C = _ds->C;
	do
	{
		for (int i = 0; i < ts->height; ++i)
		{
			double sum = 0;
			for (int j = 0; j < ts->height; ++j)
			{
				if (oldAlpha[j] == 0) continue;
				sum += ts->y[j] * oldAlpha[j] * K(ts->GetSample(i), ts->GetSample(j), ts->width);
			}
			double value = oldAlpha[i] + steps[i] - steps[i]*ts->y[i] * sum;
			if (value > C)
				value = C;
			else if (value < 0)
				value = 0.0;
			
			auto dif = value - alpha[i];
			if (dif*lastDif[i] < 0)
				steps[i] /= 2;
			lastDif[i] = dif;
			alpha[i] = value;
		}
		difAlpha = 0;
		step = 0;
		for (int i = 0; i < ts->height; ++i){
			difAlpha += lastDif[i];
			step += lastDif[i];
			oldAlpha[i] = alpha[i];
		}
		difAlpha /= ts->height;
		step /= ts->height;

		Logger::ClassifyProgress(count, step, 0, difAlpha);

		count++;
	} while ((abs(difAlpha) > Precision && count < MaxIterations) || count <= 1);
	int nSupportVectors = 0;
	for (int i = 0; i < ts->height; ++i){
		if (alpha[i] > 0)
			nSupportVectors++;
	}
	Logger::Stats("nSupportVectors", nSupportVectors);
	Logger::FunctionEnd();
}

void SequentialSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	Logger::FunctionStart("Test");
	auto start = clock();
	for (auto i = 0; i < vs->height; ++i)
	{
		int classifiedY = Classify(ts,vs->GetSample(i));
		if (classifiedY == vs->y[i]){
			vs->nCorrect++;
		}
	}
	Logger::Stats("AverageClassificationTime ", (clock() - start) / vs->height);
	auto percentageCorrect = static_cast<double>(vs->nCorrect) / vs->height;
	Logger::Percentage(vs->nCorrect, vs->height, percentageCorrect);
	Logger::FunctionEnd();
}

double SequentialSvm::K(double* x, double* y, int size)
{
	double sum = 0;
	double product;
	for (int i = 0; i < size; ++i)
	{
		product = x[i] - y[i];
		product *= product;
		sum += product;
	}
	return exp(-g*sum);
}