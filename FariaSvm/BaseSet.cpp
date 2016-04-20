#include "stdafx.h"
#include "BaseSet.h"


BaseSet::BaseSet()
{
}

BaseSet::~BaseSet()
{
	if (x != nullptr)
		delete[] x;

	if (y != nullptr)
		delete[] y;
}

void BaseSet::Init(int height, int width)
{
	if (height != this->height){
		this->height = height;
		this->width = width;

		if (x != nullptr)
			delete[] x;
		x = new double[height*width];

		if (y != nullptr)
			delete[] y;
		y = new double[height];
	}
	last = 0;
};

void BaseSet::PushSample(vector<double> x, double y)
{
	for (int j = 0; j < x.size(); j++)
		this->x[last*width + j] = x[j];
	for (int j = x.size(); j <width; j++)
		this->x[last*width + j] = 0;
	this->y[last] = y;
	last++;
}
