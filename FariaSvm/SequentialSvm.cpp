#include "stdafx.h"
#include "SequentialSvm.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"
using namespace std;

SequentialSvm::SequentialSvm(int argc, char** argv, DataSet *ds)
	: BaseSvm(argc, argv, ds)
{
	Logger::instance()->Stats("Blocks", 0);
	Logger::instance()->Stats("Threads", 0);
}

SequentialSvm::~SequentialSvm()
{
}

int SequentialSvm::Classify(TrainingSet *ts, double* sample)
{
	auto m = Logger::instance()->StartMetric("Classify");
	auto sum = 0.0;
	for (auto i = 0; i < ts->height; ++i)
	{
		if (ts->alpha[i] == 0) continue;
		sum += ts->alpha[i] * ts->y[i] * K(ts->GetSample(i), sample,ts->width);
	}
	auto sign = sum - ts->b;
	m->Stop();
	if (sign > Precision)
		return 1;
	if (sign < Precision)
		return -1;
	return 0;
}

void SequentialSvm::Train(TrainingSet *ts)
{
	auto m = Logger::instance()->StartMetric("Train");
	Logger::instance()->FunctionStart("Train");
	double* alpha = ts->alpha;
	vector<double> oldAlpha;
	int samples = _ds->nSamples;
	for (int i = 0; i < ts->height; ++i){
		alpha[i] = _initialStep;
		oldAlpha.push_back(_initialStep);
	}
	int count = 0;
	double *lastDif = (double*)malloc(ts->height*sizeof(double));
	for (int i = 0; i < ts->height; i++)
		lastDif[i] = 0;
	double difAlpha;
	double *steps = (double*)malloc(ts->height*sizeof(double));
	for (int i = 0; i < ts->height; i++)
		steps[i] = _initialStep;
	double step;
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
			step += steps[i];
			oldAlpha[i] = alpha[i];
		}
		difAlpha /= ts->height;
		step /= ts->height;

		Logger::instance()->ClassifyProgress(count, step, 0, difAlpha);

		count++;
	} while ((abs(difAlpha) > Precision && count < MaxIterations) || count <= 1);
	int nSupportVectors = 0;
	for (int i = 0; i < ts->height; ++i){
		if (alpha[i] > 0)
			nSupportVectors++;
	}
	free(lastDif);
	free(steps);
	Logger::instance()->Stats("nSupportVectors", nSupportVectors);
	Logger::instance()->FunctionEnd("Train");
	m->Stop();
}

void SequentialSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	auto m = Logger::instance()->StartMetric("Test");
	Logger::instance()->FunctionStart("Test");
	auto start = clock();
	for (auto i = 0; i < vs->height; ++i)
	{
		int classifiedY = Classify(ts, vs->GetSample(i));
		if (classifiedY == vs->y[i])
			vs->nCorrect++;
		else if (classifiedY<0)
			vs->nNegativeWrong++;
		else if (classifiedY>0)
			vs->nPositiveWrong++;
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
	m->Stop();
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