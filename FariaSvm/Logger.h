#pragma once
#include <string>
#include <vector>
using namespace std;
struct FunctionTimer
{
	string name;
	unsigned int start;
};
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
	LoggerType _type;
	unsigned int _programStart;
	unsigned int _functionStart;
	string _currentFunction;
	~Logger();
	void Init(int argc, char** argv);
	void Seed(unsigned int seed);
	void Fold(int i);
	void Percentage(double totalCorrect, double totalSamples, double averagePercentageCorrect, string title = "");
	void End();
	void Error(exception exception);
	void FunctionStart(string functionName);
	void FunctionEnd();
	void ClassifyProgress(int count, double step, double lastDif, double difAlpha);
	void Stats(string statName, long stat);
	void Stats(string statName, int stat);
	void Stats(string statName, unsigned int stat);
	void Stats(string statName, double stat);
	void Stats(string statName, string stat);
	void Line(string s);
	string FormatClock(int milliseconds);
	string FormatClock();
private:
	static Logger *s_instance;
	Logger();
};

