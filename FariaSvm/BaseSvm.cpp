#include "stdafx.h"
#include "BaseSvm.h"
#include "Settings.h"
#include "SequentialSvm.h"
#include "ParallelSvm.cuh"


class ParallelSvm;

BaseSvm::BaseSvm(DataSet &ds)
{
	_ds = ds;

	Settings::instance()->GetDouble("precision", Precision);

	double gama;
	Settings::instance()->GetDouble("gamma", gama, _ds.Gama);
	g = gama;
	string arg;
	arg = Settings::instance()->GetString("stepMode");
	isMultiStep = arg == "m";

	arg = Settings::instance()->GetString("stochastic");
	isStochastic = arg == "t";

	Settings::instance()->GetDouble("constraint", C, _ds.C);

	Settings::instance()->GetDouble("step", _initialStep);

	Settings::instance()->GetUnsigned("maxIterations", MaxIterations);
}

BaseSvm* BaseSvm::GenerateSvm(DataSet ds, string arg)
{
	if (arg.empty())
		arg = Settings::instance()->GetString("svm");
	if (arg == "p")
		return new ParallelSvm(ds);
	return new SequentialSvm(ds);
}

void BaseSvm::Train(TrainingSet *ts)
{
	throw(exception("Not Implemented."));
}

void BaseSvm::Test(TrainingSet *ts, ValidationSet *vs)
{
	throw(exception("Not Implemented."));
}

BaseSvm::~BaseSvm()
{
}