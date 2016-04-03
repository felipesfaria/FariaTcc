#pragma once
#include <string>
#include <vector>
#include "Timer.h"
#include <map>
#include <fstream>
#include "Settings.h"
using namespace std;

class Logger
{
public:
	static Logger *instance();
	~Logger();
	void Seed(unsigned int seed);
	void Fold(int i);
	void Percentage(double totalCorrect, double totalSamples, double averagePercentageCorrect, string title = "");
	void End();
	void Error(exception exception);
	void FunctionStart(string functionName);
	void FunctionEnd(string functionName);
	TimeMetric* StartMetric(string name);
	void ClassifyProgress(int count, double step, double lastDif, double difAlpha);
	void Stats(string statName, long stat);
	void Stats(string statName, int stat);
	void Stats(string statName, unsigned int stat);
	void Stats(string statName, double stat);
	void Stats(string statName, string stat);
	void Line(string s);
	void LogSettings();
private:
	fstream logFile;
	map<string, Timer*> FunctionTimers;
	map<string, Metric*> Metrics;
	map<string, string> StatsMap;
	string FormatClock(unsigned milliseconds, bool addTab = true);
	string FormatClock();
	unsigned int _programStart;
	static Logger *s_instance;
	Logger();
};

