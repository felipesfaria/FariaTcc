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
