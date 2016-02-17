#include "stdafx.h"
#include "Logger.h"
#include <locale>
#include <sstream>
#include <iostream>
using namespace std;

Logger::Logger()
{
}

void Logger::Init(int argc, char** argv)
{
	_functionStart = -1;
	_programStart = clock();
	if (cmdOptionExists(argv, argv + argc, "lv"))
		_type = VERBOSE;
	else if (cmdOptionExists(argv, argv + argc, "lcsv"))
		_type = CSV;
	else
		_type = DISABLED;

	switch (_type)
	{
	case VERBOSE:
		cout << FormatClock() << "Program Started" << endl;;
		break;
	case CSV:
		cout << "DataSet;Samples;TrainingSize;TrainingSpeed;TestSize;TestSize;CompleteTime;Folds;Precision;" << endl;;
		break;
	default:
		break;
	}
}

Logger::~Logger()
{
}

void Logger::Seed(unsigned seed)
{
	switch (_type)
	{
	case VERBOSE:
		cout << FormatClock() << "seed: " << seed << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}

void Logger::Fold(int i)
{
	if (_type != VERBOSE) return;
	cout << FormatClock() << "Fold: " << i << endl;
}

void Logger::Error(exception exception)
{
	cout << FormatClock() << "Fatal error ocurred: " << exception.what() << endl;
}

void Logger::FunctionStart(string functionName)
{
	if (!_currentFunction.empty())
		throw(new exception("Called Logger.FunctionStart before Logger.FunctionEnd."));
	_currentFunction = functionName;
	_functionStart = clock();
	switch (_type)
	{
	case VERBOSE:
		cout << FormatClock() << _currentFunction << " starting..." << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}
void Logger::FunctionEnd()
{
	if (_currentFunction.empty())
		throw(new exception("Called Logger.FunctionEnd before Logger.FunctionStart."));

	int end = clock();
	switch (_type)
	{
	case VERBOSE:
		cout << FormatClock() << _currentFunction << " finished in " << FormatClock(end - _functionStart) << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
	_currentFunction.clear();
}

void Logger::ClassifyProgress(int count, double step, double lastDif, double difAlpha)
{
	switch (_type)
	{
	case VERBOSE:
		cout << FormatClock() << "Iteration: " << count << "\tstep: " << step << "\tlastDif:" << lastDif << "\tdifAlpha:" << difAlpha << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}

void Logger::Stats(string statName, int stat)
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
	switch (_type)
	{
	case VERBOSE:
		cout << statName << ": " << stat << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}

void Logger::Line(string s)
{
	switch (_type)
	{
	case VERBOSE:
		cout << s << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}

void Logger::End()
{
	int end;
	switch (_type)
	{
	case VERBOSE:
		end = clock();
		cout << FormatClock() << "Program Finished in " << FormatClock(end - _programStart) << endl;
		cout << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}

void Logger::Percentage(double correct, double total, double percentage, string title)
{
	switch (_type)
	{
	case VERBOSE:
		cout << FormatClock() << title << "Percentage correct: " << correct << "/" << total << " = " << percentage*100.0 << "%" << endl;
		break;
	case CSV:

		break;
	default:
		break;
	}
}

bool Logger::cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return find(begin, end, option) != end;
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

