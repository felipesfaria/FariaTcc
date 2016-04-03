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
	unsigned GetAverage() const;
	string GetName() const;
protected:
	string name;
	unsigned int acumulated = 0;
	unsigned int count = 0;
};

class TimeMetric : 
	public Metric
{
public:
	TimeMetric(string name);
	void Start();
	void Stop();
private:
	bool isRunning = false;
	unsigned int start = 0;
};