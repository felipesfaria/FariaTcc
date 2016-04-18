#include "stdafx.h"
#include "SequentialSvm.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"
#include "CudaKernels.cuh"
using namespace std;

SequentialSvm::SequentialSvm(DataSet &ds)
	: BaseSvm(ds)
{
	Logger::instance()->Stats("Blocks", 0);
	Logger::instance()->Stats("Threads", 0);
}

SequentialSvm::~SequentialSvm()
{
}

int SequentialSvm::Classify(TrainingSet *ts, ValidationSet* vs, unsigned vIndex)
{
	auto m = Logger::instance()->StartMetric("Classify");
	auto x = ts->x;
	auto y = ts->y;
	auto vX = vs->x;
	auto sum = 0.0;
	auto height = ts->height;
	auto width = ts->width;
	auto alpha = ts->alpha;
	for (auto i = 0; i < height; ++i)
		sum += alpha[i] * y[i] * gaussKernel(x, i, vX, vIndex, width, g);
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
	auto newAlpha = ts->alpha;
	auto height = ts->height;
	auto x = ts->x;
	auto y = ts->y;
	auto width = ts->width;
	double* alpha = new double[height];
	for (int i = 0; i < height; ++i){
		newAlpha[i] = _initialStep;
		alpha[i] = _initialStep;
	}
	int count = 0;

	double *steps;
	double *oldDifs;
	if (isMultiStep){
		oldDifs = (double*)malloc(height*sizeof(double));
		for (int i = 0; i < height; i++)
			oldDifs[i] = 0.0;

		steps = (double*)malloc(height*sizeof(double));
		for (int i = 0; i < height; i++)
			steps[i] = _initialStep;
	}
	else{
		oldDifs = (double*)malloc(sizeof(double));
		oldDifs[0] = 0.0;

		steps = (double*)malloc(sizeof(double));
		steps[0] = _initialStep;
	}

	double avgDif;
	double avgStep;
	do
	{
		avgDif = 0;
		avgStep = 0.0;
		for (int i = 0; i < height; ++i)
		{
			double sum = 0;
			for (int j = 0; j < height; ++j)
				sum += y[j] * alpha[j] * gaussKernel(x, j, x, i, width, g);

			double value;
			if (isMultiStep)
				value = calcAlpha(alpha[i], sum, y[i], steps[i], C);
			else
				value = calcAlpha(alpha[i], sum, y[i], steps[0], C);

			auto newDif = value - alpha[i];

			if (isMultiStep)
				updateStep(steps[i], oldDifs[i], newDif);

			avgDif += newDif;
			if (isStochastic)
				alpha[i] = value;
			else
				newAlpha[i] = value;
		}

		if (!isStochastic)
			for (int i = 0; i < height; ++i)
				alpha[i] = newAlpha[i];

		avgDif /= height;

		if (!isMultiStep){
			updateStep(steps[0], oldDifs[0], avgDif);
			avgStep = steps[0];
		}
		else{
			for (int i = 0; i < height; ++i){
				avgStep += steps[i];
				avgStep /= height;
			}
		}

		count++;

		Logger::instance()->TrainingProgress(count, avgStep, avgDif);

	} while (abs(avgDif) > Precision && count < MaxIterations);

	Logger::instance()->AddIntMetric("Iterations", count);

	free(oldDifs);
	free(steps);
	free(alpha);
	m->Stop();
}

void SequentialSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	auto m = Logger::instance()->StartMetric("Test");
	for (auto i = 0; i < vs->height; ++i)
	{
		int classifiedY = Classify(ts, vs, i);
		vs->Validate(i, classifiedY);
	}
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