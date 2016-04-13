#include "stdafx.h"
#include "SequentialSvm.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"
#include "CudaKernels.cuh"
using namespace std;

SequentialSvm::SequentialSvm(DataSet *ds)
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
	auto alpha = ts->alpha;
	auto height = ts->height;
	auto x = ts->x;
	auto y = ts->y;
	auto width = ts->width;
	double* oldAlpha = new double[height];
	for (int i = 0; i < height; ++i){
		alpha[i] = _initialStep;
		oldAlpha[i] = _initialStep;
	}
	int count = 0;
	double *lastDif = (double*)malloc(height*sizeof(double));
	for (int i = 0; i < height; i++)
		lastDif[i] = 0;
	double difAlpha;
	double *steps = (double*)malloc(height*sizeof(double));
	for (int i = 0; i < height; i++)
		steps[i] = _initialStep;
	double step;
	do
	{
		for (int i = 0; i < height; ++i)
		{
			double sum = 0;
			for (int j = 0; j < height; ++j)
			{
				sum += y[j] * oldAlpha[j] * gaussKernel(x,j,x,i,width,g);
			}
			double value = calcAlpha(oldAlpha, sum, y, steps, C, i);
			
			auto dif = value - alpha[i];
			if (dif*lastDif[i] < 0)
				steps[i] /= 2;
			lastDif[i] = dif;
			alpha[i] = value;
		}
		difAlpha = 0;
		step = 0;
		for (int i = 0; i < height; ++i){
			difAlpha += lastDif[i];
			step += steps[i];
			oldAlpha[i] = alpha[i];
		}
		difAlpha /= height;
		step /= height;

		count++;

		Logger::instance()->TrainingProgress(count, step, 0, difAlpha);

	} while ((abs(difAlpha) > Precision && count < MaxIterations));

	Logger::instance()->AddIntMetric("Iterations", count);

	free(lastDif);
	free(steps);
	free(oldAlpha);
	m->Stop();
}

void SequentialSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	auto m = Logger::instance()->StartMetric("Test");
	for (auto i = 0; i < vs->height; ++i)
	{
		int classifiedY = Classify(ts, vs,i);
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