#include "stdafx.h"
#include "BaseSvm.h"


BaseSvm::BaseSvm()
{
	throw(new exception("Not Implemented."));
}

int BaseSvm::Classify(const DataSet& ds, int index, vector<double>& alpha, double& b)
{
	throw(new exception("Not Implemented."));
}

void BaseSvm::Train(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha, double& b)
{
	throw(new exception("Not Implemented."));
}

void BaseSvm::Test(const DataSet& ds, int validationStart, int validationEnd, vector<double>& alpha1, double& b1, int& nCorrect)
{
	throw(new exception("Not Implemented."));
}

BaseSvm::~BaseSvm()
{
}
