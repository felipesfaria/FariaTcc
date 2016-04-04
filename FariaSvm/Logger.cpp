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
	_programStart = clock();
	logFile << FormatClock()<<"\t"<< "Program Started" << endl;;
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

void Logger::Seed(unsigned seed)
{
	logFile << FormatClock()<<"\t" << "seed: " << seed << endl;
}

void Logger::Fold(int i)
{
	logFile << FormatClock()<<"\t" << "Fold: " << i << endl;
}

void Logger::Error(exception exception)
{
	logFile << FormatClock()<<"\t" << "Fatal error ocurred: " << exception.what() << endl;
}

void Logger::FunctionStart(string functionName)
{
	if (FunctionTimers.count(functionName))
		throw exception(("FunctionTimer:" + functionName + " allready started").c_str());
	FunctionTimers[functionName] = new Timer(functionName);
	logFile << FormatClock()<<"\t" << functionName << " starting..." << endl;
}
void Logger::FunctionEnd(string functionName)
{
	unsigned elapsed;
	if (!FunctionTimers.count(functionName))
		throw exception(("FunctionTimer:" + functionName + " hasn't started").c_str());
	auto timer = FunctionTimers[functionName];
	FunctionTimers.erase(functionName);
	elapsed = timer->GetElapsed();
	delete(timer);
	logFile << FormatClock()<<"\t" << functionName << " finished in " << FormatClock(elapsed) << endl;
}

TimeMetric* Logger::StartMetric(string name)
{
	if (Metrics.count(name) == 0)
		Metrics[name] = new TimeMetric(name);
	auto m = (TimeMetric*)Metrics[name];
	m->Start();
	return m;
}

void Logger::AddIntMetric(string name, unsigned value)
{
	if (Metrics.count(name) == 0)
		Metrics[name] = new IntMetric(name);
	auto m = (IntMetric*)Metrics[name];
	m->Add(value);
}

void Logger::AddDoubleMetric(string name, double value)
{
	if (Metrics.count(name) == 0)
		Metrics[name] = new DoubleMetric(name);
	auto m = (DoubleMetric*)Metrics[name];
	m->Add(value);
}

void Logger::ClassifyProgress(int count, double step, double lastDif, double difAlpha)
{
	logFile << FormatClock()<<"\t" << "Iteration: " << count << "\tstep: " << step << "\tlastDif:" << lastDif << "\tdifAlpha:" << difAlpha << endl;
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
	logFile << FormatClock()<<"\t" << statName << ": " << stat << endl;
}

void Logger::Line(string s)
{
	logFile << s << endl;
}

void Logger::LogSettings()
{
	auto settingsMap = Settings::instance()->settingsMap;

	for (auto it = settingsMap.begin(); it != settingsMap.end(); ++it){
		auto setting = (*it).second;
		auto str = setting.ToString();
		if (!str.empty())
			logFile << FormatClock()<<"\t" << "Settings."<<setting.name <<" = "<< str<< endl;
	}
}

void Logger::End()
{
	int end = clock();
	logFile << FormatClock()<<"\t" << "Program Finished in " << FormatClock(end - _programStart) << endl;
	logFile << endl;

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
			resultFile << FormatClock(tm->GetAverage())<<"\t";
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
	delete(instance());
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

