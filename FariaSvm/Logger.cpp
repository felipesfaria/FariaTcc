#include "stdafx.h"
#include "Logger.h"
#include <locale>
#include <sstream>
#include <iostream>
#include "Utils.h"
#include <fstream>
using namespace std;
Logger *Logger::s_instance = nullptr;

Logger::Logger()
{
	logFile.open("log.txt", fstream::out | fstream::trunc);
}

Logger* Logger::instance()
{
	if (!s_instance)
		s_instance = new Logger();
	return s_instance;
}

Logger::~Logger()
{
	logFile.close();
}

void Logger::Init(int argc, char** argv)
{
	_programStart = clock();
	string argument = Utils::GetComandVariable(argc, argv, "-l");
	if (argument == "v")
		_type = VERBOSE;
	else if (argument == "c")
		_type = CSV;
	else if (argument == "d" || argument == "")
		_type = DISABLED;
	else
	{
		string message = "Invalid argument for -l:" + argument;
		throw(exception(message.c_str()));
	}

	logFile << FormatClock() << "Program Started" << endl;;
}

void Logger::Seed(unsigned seed)
{
	logFile << FormatClock() << "seed: " << seed << endl;
}

void Logger::Fold(int i)
{
	if (_type != VERBOSE) return;
	logFile << FormatClock() << "Fold: " << i << endl;
}

void Logger::Error(exception exception)
{
	logFile << FormatClock() << "Fatal error ocurred: " << exception.what() << endl;
}

void Logger::FunctionStart(string functionName)
{
	auto value = FunctionTimers.find(functionName);
	if (value != FunctionTimers.end())
		throw exception(("FunctionTimer:" + functionName + " allready started").c_str());
	FunctionTimers[functionName] = new Timer(functionName);
	logFile << FormatClock() << functionName << " starting..." << endl;
}
void Logger::FunctionEnd(string functionName)
{
	unsigned elapsed;
	auto value = FunctionTimers.find(functionName);
	if (value == FunctionTimers.end())
		throw exception(("FunctionTimer:" + functionName + " hasn't started").c_str());
	auto timer = FunctionTimers[functionName];
	FunctionTimers.erase(functionName);
	elapsed = timer->GetElapsed();
	delete(timer);
	logFile << FormatClock() << functionName << " finished in " << FormatClock(elapsed) << endl;
}

void Logger::ClassifyProgress(int count, double step, double lastDif, double difAlpha)
{
	logFile << FormatClock() << "Iteration: " << count << "\tstep: " << step << "\tlastDif:" << lastDif << "\tdifAlpha:" << difAlpha << endl;
}

void Logger::Stats(string statName, long stat)
{
	std::ostringstream strs;
	strs << stat;
	Stats(statName, strs.str());
}

void Logger::Stats(string statName, int stat)
{
	std::ostringstream strs;
	strs << stat;
	Stats(statName, strs.str());
}
void Logger::Stats(string statName, unsigned int stat)
{
	std::ostringstream strs;
	strs << stat;
	Stats(statName, strs.str());
}
void Logger::Stats(string statName, double stat)
{
	std::ostringstream strs;
	strs << stat;
	Stats(statName, strs.str());
}
void Logger::Stats(string statName, string stat)
{
	logFile << FormatClock() << statName << ": " << stat << endl;
}

void Logger::Line(string s)
{
	logFile << s << endl;
}

void Logger::End()
{
	int end = clock();
	logFile << FormatClock() << "Program Finished in " << FormatClock(end - _programStart) << endl;
	logFile << endl;
}

void Logger::Percentage(double correct, double total, double percentage, string title)
{
	logFile << FormatClock() << title << "Percentage correct: " << correct << "/" << total << " = " << percentage*100.0 << "%" << endl;
}

std::string Logger::FormatClock(int milliseconds)
{
	std::stringstream ss;
	auto hours = milliseconds / (60 * 60 * CLOCKS_PER_SEC);
	milliseconds = milliseconds % (60 * 60 * CLOCKS_PER_SEC);
	auto minutes = milliseconds / (60 * CLOCKS_PER_SEC);
	milliseconds = milliseconds % (60 * CLOCKS_PER_SEC);
	auto seconds = milliseconds / (CLOCKS_PER_SEC);
	milliseconds = milliseconds % (CLOCKS_PER_SEC);
	ss << hours << ":" << minutes << ":" << seconds << ":" << milliseconds << "\t";
	return ss.str();
}

std::string Logger::FormatClock()
{
	return FormatClock(clock());
}

