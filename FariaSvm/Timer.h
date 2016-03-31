#pragma once
#include <string>
using namespace std;
class Timer
{
public:
	Timer(string name);
	~Timer();
	unsigned int GetElapsed() const;
	string GetName() const;
private:
	string name;
	unsigned int start;
};

class Metric
{
public:
	Metric(string name);
	~Metric();
	void Start();
	void Stop();
	double GetAverage() const;
	string GetName() const;
private:
	bool isRunning = false;
	string name;
	unsigned int start;
	unsigned int acumulated;
	unsigned int count = 0;
};
