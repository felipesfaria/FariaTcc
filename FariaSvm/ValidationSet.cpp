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

ValidationSet::ValidationSet()
{
}


ValidationSet::~ValidationSet()
{
}
