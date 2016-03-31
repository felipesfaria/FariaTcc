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
