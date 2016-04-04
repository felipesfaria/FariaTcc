#include "stdafx.h"
#include "ValidationSet.h"


void ValidationSet::Init(int height, int width)
{
	nCorrect = 0;
	nPositiveWrong = 0;
	nNegativeWrong = 0;
	nNullWrong = 0;
	BaseInit(height, width);
}

double ValidationSet::GetPercentage()
{
	return nCorrect * 100.0 / height;
}

void ValidationSet::Validate(int i, double classifiedY)
{
	if (classifiedY == y[i])
		nCorrect++;
	else if (classifiedY<0)
		nNegativeWrong++;
	else if (classifiedY>0)
		nPositiveWrong++;
	else
		nNullWrong++;
}

ValidationSet::ValidationSet()
{
}


ValidationSet::~ValidationSet()
{
}
