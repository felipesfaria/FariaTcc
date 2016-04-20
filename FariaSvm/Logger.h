#pragma once
#include <string>
#include <vector>
#include "Timer.h"
#include <map>
#include <fstream>
#include "Settings.h"
using namespace std;
namespace FariaSvm{
	class Logger
	{
	public:
		unsigned NONE = 0;
		unsigned ERRORS = 1;
		unsigned RESULTS = 2;
		unsigned ALL = 3;
		static Logger *instance();
		static void Delete();
		~Logger();
		void End();
		void Error(exception exception);
		TimeMetric* StartMetric(string name);
		void StopMetric(TimeMetric* m);
		void AddIntMetric(string name, unsigned value);
		void AddDoubleMetric(string name, double value);
		void TrainingProgress(int count, double step, double difAlpha);
		void ClassifyingProgress(int count, double step, double lastDif, double difAlpha);
		void Stats(string statName, string stat);
		void Line(string s);
		void LogSettings();
	private:
		unsigned _type;
		fstream logFile;
		fstream errorFile;
		map<string, Timer*> FunctionTimers;
		map<string, Metric*> Metrics;
		map<string, string> StatsMap;
		string FormatClock(unsigned milliseconds);
		string FormatClock();
		unsigned int _programStart;
		static Logger *s_instance;
		Logger();
	};
}