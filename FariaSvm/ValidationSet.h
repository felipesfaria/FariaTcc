#pragma once
#include "BaseSet.h"

class ValidationSet: public BaseSet
{
public:
	int nCorrect = 0;
	int nPositiveWrong = 0;
	int nNegativeWrong = 0;
	int nNullWrong = 0;
	void Init(int height, int width) override;
	double GetPercentage() const;
	void Validate(int i, double classifiedY);
	ValidationSet();
	~ValidationSet();
};

