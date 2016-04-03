#include "stdafx.h"
#include "BaseSvm.h"
#include "Settings.h"


BaseSvm::BaseSvm(int argc, char** argv, DataSet *ds)
{
	_ds = ds;

	Settings::instance()->GetDouble("precision", Precision);

	double gama;
	Settings::instance()->GetDouble("gamma", gama, _ds->Gama);
	g = 1 / (2 * gama*gama);

	Settings::instance()->GetDouble("constraint", C, _ds->C);

	Settings::instance()->GetDouble("step", _initialStep);

	Settings::instance()->GetUnsigned("maxIterations", MaxIterations);
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
