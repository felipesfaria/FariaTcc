#pragma once
#include <vector>

using namespace std;

class BaseSet
{
public:
	double* x;
	double* y;
	int width = 0;
	int height = 0;
	int last = 0;
	bool initialised = false;
	BaseSet();
	~BaseSet();
	void Init(int height, int width);
	void BaseInit(int height, int width);
	void PushSample(vector<double> x, double y);
};

