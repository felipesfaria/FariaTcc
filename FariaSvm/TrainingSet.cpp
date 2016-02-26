#include "stdafx.h"
#include "TrainingSet.h"

void TrainingSet::Init(int height, int width)
{
	if (height != this->height){
		if (initialised)
			free(alpha);
		alpha = (double*)malloc(height*sizeof(double));
		b = 0.0;
	}
	BaseInit(height, width);
}

TrainingSet::~TrainingSet()
{
	if (initialised)
		free(alpha);
}
