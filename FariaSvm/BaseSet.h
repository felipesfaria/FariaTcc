#pragma once
#include <vector>
#include <memory>

using namespace std;

class BaseSet
{
public:
	double* x = nullptr;
	double* y = nullptr;
	int width = 0;
	int height = 0;
	int last = 0;
	BaseSet();
	virtual ~BaseSet();
	virtual void Init(int height, int width);
	void PushSample(vector<double> x, double y);
};

