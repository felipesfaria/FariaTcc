#include "stdafx.h"
#include "TrainingSet.h"
#include "Settings.h"

void TrainingSet::Init(int height, int width)
{
	if (height != this->height){
		if (initialised)
			free(alpha);
		alpha = (double*)malloc(height*sizeof(double));
		b = 0.0;
	}
	BaseSet::Init(height, width);
}

unsigned TrainingSet::CountSupportVectors() const
{
	unsigned count = 0;
	for (int i = 0; i < height; ++i){
		if (alpha[i] != 0)
			count++;
	}
	return count;
}

TrainingSet::~TrainingSet()
{
	if (initialised)
		free(alpha);
}
