#include "stdafx.h"
#include "BaseSvm.h"


BaseSvm::BaseSvm()
{
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
