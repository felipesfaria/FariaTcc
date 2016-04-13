#include "stdafx.h"
#include "BaseSet.h"


BaseSet::BaseSet()
{
}

BaseSet::~BaseSet()
{
	if (initialised){
		free(x);
		free(y);
	}
}

void BaseSet::Init(int height, int width)
{
	if (height != this->height){
		this->height = height;
		this->width = width;
		if (initialised){
			free(x);
			free(y);
		}
		x = (double*)malloc(height*width*sizeof(double));
		y = (double*)malloc(height*sizeof(double));
	}
	last = 0;
	initialised = true;
};

void BaseSet::PushSample(vector<double> x, double y)
{
	for (int j = 0; j < width; j++)
		this->x[last*width + j] = x[j];
	this->y[last] = y;
	last++;
}

double* BaseSet::GetSample(int i)
{
	return x + i*width;
}
