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
	_programStart = clock();
	auto t = Settings::instance()->GetString("log");
	if (t == "a")
		_type = ALL;
	else if (t == "e")
		_type = ERRORS;
	else if (t == "n")
		_type = NONE;
	else
		_type = RESULTS;

	logFile.open("log.txt", fstream::out | fstream::trunc);


	if (_type < ALL) return;
	logFile << FormatClock() << "\t" << "Program Started" << endl;
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

void Logger::Error(exception exception)
{
	cout << FormatClock() << "\t" << "Fatal error ocurred: " << exception.what() << endl;

	if (_type < ERRORS) return;
	logFile << FormatClock() << "\t" << "Fatal error ocurred: " << exception.what() << endl;
}

TimeMetric* Logger::StartMetric(string name)
{
	if (Metrics.count(name) == 0)
		Metrics[name] = new TimeMetric(name);
	auto m = (TimeMetric*)Metrics[name];
	m->Start();
	if (_type <= ALL)
		Line(name);
	return m;
}

void Logger::AddIntMetric(string name, unsigned value)
{
	if (Metrics.count(name) == 0)
		Metrics[name] = new IntMetric(name);
	auto m = (IntMetric*)Metrics[name];
	m->Add(value);
	if (_type <= ALL)
		Line(name+": "+to_string(value));
}

void Logger::AddDoubleMetric(string name, double value)
{
	if (Metrics.count(name) == 0)
		Metrics[name] = new DoubleMetric(name);
	auto m = (DoubleMetric*)Metrics[name];
	m->Add(value);
	if (_type <= ALL)
		Line(name + ": " + to_string(value));
}

void Logger::TrainingProgress(int count, double step, double lastDif, double difAlpha)
{
	if (_type < ALL) return;
	logFile << FormatClock() << "\t" << "Iteration: " << count << "\tstep: " << step << "\tlastDif:" << lastDif << "\tdifAlpha:" << difAlpha << endl;
}

void Logger::ClassifyingProgress(int count, double step, double lastDif, double difAlpha)
{
	if (_type < ALL) return;
	logFile << FormatClock() << "\t" << "Iteration: " << count << "\tstep: " << step << "\tlastDif:" << lastDif << "\tdifAlpha:" << difAlpha << endl;
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
	StatsMap[statName] = stat;

	if (_type < ALL) return;
	logFile << FormatClock() << "\t" << statName << ": " << stat << endl;
}

void Logger::Line(string s)
{
	if (_type < ALL) return;

	logFile << FormatClock() << "\t" << s << endl;
}

void Logger::LogSettings()
{
	if (_type < ALL) return;
	auto settingsMap = Settings::instance()->settingsMap;

	for (auto it = settingsMap.begin(); it != settingsMap.end(); ++it){
		auto setting = (*it).second;
		auto str = setting.ToString();
		if (!str.empty())
			logFile << FormatClock() << "\t" << "Settings." << setting.name << " = " << str << endl;
	}
}

void Logger::End()
{
	if (_type < RESULTS) return;

	auto settingsMap = Settings::instance()->settingsMap;

	fstream resultFile;
	stringstream headerStream;
	for (auto it = StatsMap.begin(); it != StatsMap.end(); ++it)
		headerStream << it->first << "\t";

	for (auto it = settingsMap.begin(); it != settingsMap.end(); ++it){
		if (it->second.type != Setting::DOUBLE
			&& it->second.type != Setting::UNSIGNED
			&& it->second.type != Setting::STRING)
			continue;
		headerStream << it->first << "\t";
	}

	for (auto it = Metrics.begin(); it != Metrics.end(); ++it)
		headerStream << it->first << "\t";
	auto header = headerStream.str();

	resultFile.open("results.txt", fstream::in);
	string         line;
	getline(resultFile, line);

	resultFile.close();
	if (headerStream.str() != line){
		resultFile.open("results.txt", fstream::out | fstream::trunc);
		resultFile << headerStream.str();
	}
	else
		resultFile.open("results.txt", fstream::out | fstream::app);
	resultFile << endl;

	for (auto it = StatsMap.begin(); it != StatsMap.end(); ++it)
		resultFile << it->second << "\t";

	for (auto it = settingsMap.begin(); it != settingsMap.end(); ++it)
	{
		auto setting = it->second;
		string str = setting.ToString();
		if (!str.empty())
			resultFile << str << "\t";
	}

	for (auto it = Metrics.begin(); it != Metrics.end(); ++it){
		Metric* metric = it->second;
		if (auto tm = dynamic_cast<TimeMetric*>(metric))
		{
			resultFile << FormatClock(tm->GetAverage()) << "\t";
		}
		else if (auto im = dynamic_cast<IntMetric*>(metric))
		{
			resultFile << im->GetAverage() << "\t";
		}
		else if (auto dm = dynamic_cast<DoubleMetric*>(metric))
		{
			resultFile << dm->GetAverage() << "\t";
		}
	}

	resultFile.close();

	if (_type < ALL) return;

	logFile << FormatClock() << "\t" << "Program Finished in " << FormatClock(clock() - _programStart) << endl;
	logFile << endl;
}

std::string Logger::FormatClock(unsigned milliseconds)
{
	std::stringstream ss;
	auto hours = milliseconds / (60 * 60 * CLOCKS_PER_SEC);
	milliseconds = milliseconds % (60 * 60 * CLOCKS_PER_SEC);
	auto minutes = milliseconds / (60 * CLOCKS_PER_SEC);
	milliseconds = milliseconds % (60 * CLOCKS_PER_SEC);
	auto seconds = milliseconds / (CLOCKS_PER_SEC);
	milliseconds = milliseconds % (CLOCKS_PER_SEC);
	ss << hours << ":" << minutes << ":" << seconds << ":" << milliseconds;
	return ss.str();
}

std::string Logger::FormatClock()
{
	return FormatClock(clock());
}

