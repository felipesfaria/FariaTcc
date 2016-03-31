#include "stdafx.h"
#include "BaseSvm.h"


BaseSvm::BaseSvm(int argc, char** argv, DataSet *ds)
{
	_ds = ds;

	auto arg = Utils::GetComandVariable(argc, argv, "-p");
	if (!Utils::TryParseDouble(arg, Precision))
		Precision = 1e-15;
	Logger::instance()->Stats("Precision", Precision);
	
	double gama;
	arg = Utils::GetComandVariable(argc, argv, "-g");
	if (!Utils::TryParseDouble(arg, gama))
		gama = _ds->Gama;
	Logger::instance()->Stats("Gama", gama);
	g = 1 / (2 * gama*gama);

	arg = Utils::GetComandVariable(argc, argv, "-st");
	if (!Utils::TryParseDouble(arg, Step))
		Step = 1;
	Logger::instance()->Stats("Step", Step);
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
