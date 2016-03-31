#pragma once
#include <string>
#include <vector>
#include "Timer.h"
#include <map>
using namespace std;

class Logger
{
public:
	enum LoggerType
	{
		DISABLED,
		VERBOSE,
		CSV
	};
	static Logger *instance();
	~Logger();
	void Init(int argc, char** argv);
	void Seed(unsigned int seed);
	void Fold(int i);
	void Percentage(double totalCorrect, double totalSamples, double averagePercentageCorrect, string title = "");
	void End();
	void Error(exception exception);
	void FunctionStart(string functionName);
	void FunctionEnd(string functionName);
	void ClassifyProgress(int count, double step, double lastDif, double difAlpha);
	void Stats(string statName, long stat);
	void Stats(string statName, int stat);
	void Stats(string statName, unsigned int stat);
	void Stats(string statName, double stat);
	void Stats(string statName, string stat);
	void Line(string s);
private:
	map<string,Timer*> FunctionTimers;
	string FormatClock(int milliseconds);
	string FormatClock();
	unsigned int _programStart;
	LoggerType _type;
	static Logger *s_instance;
	Logger();
};

