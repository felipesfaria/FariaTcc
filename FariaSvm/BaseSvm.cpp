#include "stdafx.h"
#include "BaseSvm.h"


BaseSvm::BaseSvm(int argc, char** argv, DataSet *ds)
{
	_ds = ds;

	auto arg = Utils::GetComandVariable(argc, argv, "-p");
	if (!Utils::TryParseDouble(arg, Precision))
		Precision = 1e-12;
	Logger::Stats("Precision", Precision);
	
	double gama;
	arg = Utils::GetComandVariable(argc, argv, "-g");
	if (!Utils::TryParseDouble(arg, gama))
		gama = _ds->Gama;
	Logger::Stats("Gama", gama);
	g = 1 / (2 * gama*gama);

	arg = Utils::GetComandVariable(argc, argv, "-st");
	if (!Utils::TryParseDouble(arg, Step))
		Step = 1e-9;
	Logger::Stats("Step", Step);
}

int BaseSvm::Classify(int index, vector<double>& alpha, double& b)
{
	throw(exception("Not Implemented."));
}

void BaseSvm::Train(int validationStart, int validationEnd, vector<double>& alpha, double& b)
{
	throw(exception("Not Implemented."));
}

void BaseSvm::Test(int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect)
{
	throw(exception("Not Implemented."));
}

BaseSvm::~BaseSvm()
{
}
