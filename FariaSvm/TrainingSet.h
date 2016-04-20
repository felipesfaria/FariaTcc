#pragma once
#include "BaseSet.h"

class TrainingSet : public BaseSet
{
public:
	double* alpha = nullptr;
	double b = 0.0;

	void Init(int height, int width) override;
	unsigned CountSupportVectors() const;
	~TrainingSet();
};

