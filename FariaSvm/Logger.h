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
	LoggerType static _type;
	unsigned static int _programStart;
	unsigned static int _functionStart;
	string static _currentFunction;
	Logger();
	~Logger();
	void static Init(int argc, char** argv);
	void static Seed(unsigned int seed);
	void static Fold(int i);
	void static Percentage(double totalCorrect, double totalSamples, double averagePercentageCorrect, string title = "");
	void static End();
	void static Error(exception exception);
	void static FunctionStart(string functionName);
	void static FunctionEnd();
	void static ClassifyProgress(int count, double step, double lastDif, double difAlpha);
	void static Stats(string statName, long stat);
	void static Stats(string statName, int stat);
	void static Stats(string statName, unsigned int stat);
	void static Stats(string statName, double stat);
	void static Stats(string statName, string stat);
	void static Line(string s);
	string static FormatClock(int milliseconds);
	string static FormatClock();
private:
};

