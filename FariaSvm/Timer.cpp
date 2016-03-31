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

unsigned Metric::GetAverage() const
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

void Metric::Start()
{
	if (isRunning)
		throw new exception("Tried to start Metric when metric was allready running.");
	isRunning = true;
	start = clock();
}

void Metric::Stop()
{
	if (!isRunning)
		throw new exception("Tried to start Metric when metric was allready stopped.");
	isRunning = false;
	acumulated += clock() - start;
	count++;
}