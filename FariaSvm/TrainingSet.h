#pragma once
#include "BaseSet.h"

class TrainingSet : public BaseSet
{
public:
	double* alpha;
	double b;

	void Init(int height, int width);
	unsigned CountSupportVectors();
	~TrainingSet();
};

