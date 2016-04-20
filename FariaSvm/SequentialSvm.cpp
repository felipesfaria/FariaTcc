#include "stdafx.h"
#include "SequentialSvm.h"
#include "Logger.h"
#include <locale>
#include "Utils.h"
#include "CudaKernels.cuh"
using namespace std;
using namespace FariaSvm;

SequentialSvm::SequentialSvm(shared_ptr<DataSet> ds)
	: BaseSvm(ds)
{
	Logger::instance()->Stats("Blocks", to_string(0));
	Logger::instance()->Stats("Threads", to_string(0));
}

SequentialSvm::~SequentialSvm()
{
}

int SequentialSvm::Classify(const TrainingSet& ts, const ValidationSet& vs, const int vIndex)
{
	auto m = Logger::instance()->StartMetric("Classify");
	auto sum = 0.0;
	for (auto i = 0; i < ts.height; ++i)
		sum += ts.alpha[i] * ts.y[i] * gaussKernel(ts.x, i, vs.x, vIndex, ts.width, g);
	auto value = sum - ts.b;
	Logger::instance()->StopMetric(m);
	return SignOf(value);
}

void SequentialSvm::Train(TrainingSet& ts)
{
	auto m = Logger::instance()->StartMetric("Train");
	auto alpha = ts.alpha;
	auto height = ts.height;
	auto x = ts.x;
	auto y = ts.y;
	auto width = ts.width;
	double* newAlpha = new double[height];
	for (int i = 0; i < height; ++i){
		newAlpha[i] = _initialStep;
		alpha[i] = _initialStep;
	}
	int count = 0;

	unsigned stepsSize = isMultiStep?height:1;
	auto steps = new double[stepsSize];
	for (int i = 0; i < stepsSize; i++)
		steps[i] = _initialStep;

	unsigned oldDifsSize = isMultiStep ? height : 1;;
	auto oldDifs = new double[oldDifsSize];
	for (auto i = 0; i < oldDifsSize; i++)
		oldDifs[i] = 0.0;

	double avgDif;
	double avgStep;
	do
	{
		avgDif = 0;
		for (int i = 0; i < height; ++i)
		{
			double sum = 0;
			for (int j = 0; j < height; ++j)
				sum += y[j] * alpha[j] * gaussKernel(x, j, x, i, width, g);

			auto stepIndex = isMultiStep ? i : 0;
			newAlpha[i] = calcAlpha(alpha[i], sum, y[i], steps[stepIndex], C);

			auto newDif = newAlpha[i] - alpha[i];

			if (isMultiStep)
				updateStep(steps[i], oldDifs[i], newDif);

			avgDif += newDif;
			if (isStochastic)
				alpha[i] = newAlpha[i];
		}

		if (!isStochastic)
			for (int i = 0; i < height; ++i)
				alpha[i] = newAlpha[i];

		avgDif /= height;

		if (!isMultiStep)
			updateStep(steps[0], oldDifs[0], avgDif);

		avgStep = 0.0;
		for (int i = 0; i < stepsSize; ++i)
			avgStep += steps[i];
		avgStep /= stepsSize;

		count++;

		Logger::instance()->TrainingProgress(count, avgStep, avgDif);

	} while (abs(avgDif) > Precision && count < MaxIterations);

	Logger::instance()->AddIntMetric("Iterations", count);

	free(oldDifs);
	free(steps);
	free(newAlpha);
	Logger::instance()->StopMetric(m);
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