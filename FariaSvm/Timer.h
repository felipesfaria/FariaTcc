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
	virtual ~Metric();
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
	unsigned GetAverage() const;
	void Start();
	void Stop();
private:
	bool isRunning = false;
	unsigned int start = 0;
};

class IntMetric :
	public Metric
{
public:
	double GetAverage() const;
	IntMetric(string name);
	void Add(unsigned value);
private:
};

class DoubleMetric :
	public Metric
{
public:
	double GetAverage() const;
	DoubleMetric(string name);
	void Add(double value);
private:
	double dAcumulated = 0.0;
};