#include "stdafx.h"
#include "Timer.h"
#include <ctime>

Timer::Timer(string n)
{
	name = n;
	start = clock();
}

Timer::~Timer()
{
}

unsigned Timer::GetElapsed() const
{
	return clock() - start;
}

string Timer::GetName() const
{
	return name;
}

unsigned TimeMetric::GetAverage() const
{
	return acumulated / count;
}

Metric::Metric(string name)
{
	this->name = name;
}
Metric::~Metric()
{
}

TimeMetric::TimeMetric(string name)
	: Metric(name)
{
}

void TimeMetric::Start()
{
	if (isRunning)
		throw new exception("Tried to start Metric when metric was allready running.");
	isRunning = true;
	start = clock();
}

void TimeMetric::Stop()
{
	if (!isRunning)
		throw new exception("Tried to start Metric when metric was allready stopped.");
	isRunning = false;
	acumulated += clock() - start;
	count++;
}

IntMetric::IntMetric(string name)
	:Metric(name)
{
}

double IntMetric::GetAverage() const
{
	return (double)acumulated / count;
}

void IntMetric::Add(unsigned value)
{
	acumulated += value;
	count++;
}


DoubleMetric::DoubleMetric(string name)
	:Metric(name)
{
}

double DoubleMetric::GetAverage() const
{
	return dAcumulated / count;
}

void DoubleMetric::Add(double value)
{
	dAcumulated += value;
	count++;
}