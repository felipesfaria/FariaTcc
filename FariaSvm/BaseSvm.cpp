#include "stdafx.h"
#include "BaseSvm.h"
#include "Settings.h"
#include "SequentialSvm.h"
#include "ParallelSvm.cuh"
#include "Logger.h"

using namespace FariaSvm;

class ParallelSvm;

BaseSvm::BaseSvm(shared_ptr<DataSet> ds)
{
	_ds = ds;

	Settings::instance()->GetDouble("precision", Precision);

	double gama;
	Settings::instance()->GetDouble("gamma", gama, _ds->Gama);
	g = gama;
	string arg;
	arg = Settings::instance()->GetString("stepMode");
	isMultiStep = arg == "m";

	arg = Settings::instance()->GetString("stochastic");
	isStochastic = arg == "t";

	Settings::instance()->GetDouble("constraint", C, _ds->C);

	Settings::instance()->GetDouble("step", _initialStep);

	Settings::instance()->GetUnsigned("maxIterations", MaxIterations);
}

unique_ptr<BaseSvm> BaseSvm::GenerateSvm(shared_ptr<DataSet>ds, string arg)
{
	if (arg.empty())
		arg = Settings::instance()->GetString("svm");
	if (arg == "p")
		return unique_ptr<BaseSvm>(make_unique<ParallelSvm>(ds));
	return unique_ptr<BaseSvm>(make_unique<SequentialSvm>(ds));
}

int BaseSvm::Classify(TrainingSet& ts, ValidationSet& vs, int index)
{
	throw(exception("Not Implemented."));
}

void BaseSvm::Train(TrainingSet & ts)
{
	throw(exception("Not Implemented."));
}

void BaseSvm::Test(TrainingSet & ts, ValidationSet & vs)
{
	auto m = Logger::instance()->StartMetric("Test");
	for (auto i = 0; i < vs.height; ++i)
	{
		int classifiedY = Classify(ts, vs, i);
		vs.Validate(i, classifiedY);
	}
	Logger::instance()->StopMetric(m);
}

int BaseSvm::SignOf(double value)
{
	return value > 0 ? 1 : value < 0 ? -1 : 0;
}

BaseSvm::~BaseSvm()
{
}