#pragma once
#include <string>
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
	Logger();
	void Init(int argc, char** argv);
	~Logger();
	void Seed(unsigned int seed);
	void Fold(int i);
	void Percentage(double totalCorrect, double totalSamples, double averagePercentageCorrect, string title = "");
	void End();
	void Error(exception exception);
	void FunctionStart(string functionName);
	void FunctionEnd();
	void ClassifyProgress(int count, double step, double lastDif, double difAlpha);
	void Stats(string statName, int stat);
	void Stats(string statName, double stat);
	void Stats(string statName, string stat);
	void Line(string s);
private:
	LoggerType _type;
	unsigned int _programStart;
	unsigned int _functionStart;
	string _currentFunction;
	bool cmdOptionExists(char** begin, char** end, const std::string& option);
	std::string FormatClock(int milliseconds);
	std::string FormatClock();
};

