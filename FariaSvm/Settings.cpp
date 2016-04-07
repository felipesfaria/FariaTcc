#include "stdafx.h"
#include "Settings.h"
#include <ctime>
#include "Utils.h"
#include <iostream>
#include <sstream>


Settings *Settings::s_instance = nullptr;
void Settings::ShowHelp()
{
	for (auto it = settingsMap.begin(); it != settingsMap.end(); ++it)
	{
		cout << it->second.command<<" : "<<it->second.description <<"Default is: "<< it->second.ToString()<< endl;
	}
	exit(0);
}

Settings::Settings()
{
}

string Setting::ToString()
{
	stringstream ss;
	switch (type)
	{
	case STRING:
		ss << sValue;
		break;
	case DOUBLE:
		ss << dValue;
		break;
	case UNSIGNED:
		ss << uValue;
		break;
	}
	return ss.str();
}

Settings* Settings::instance()
{
	if (!s_instance)
		s_instance = new Settings();
	return s_instance;
}
void Settings::Init(int argc, char** argv)
{
	Setting seed;
	seed.name = "seed";
	seed.isSet = false;
	seed.command = "-sd";
	seed.description = "Seed used for random number generator. Default is: time(nullptr)";
	seed.type = Setting::UNSIGNED;
	seed.uValue = time(nullptr);
	settingsMap[seed.name] = seed;

	Setting fold;
	fold.name = "folds";
	fold.isSet = false;
	fold.command = "-f";
	fold.description = "Folds used in cross validation. Default is 10";
	fold.type = Setting::UNSIGNED;
	fold.uValue = 10;
	settingsMap[fold.name] = fold;

	Setting threadsPerBlock;
	threadsPerBlock.name = "threadsPerBlock";
	threadsPerBlock.isSet = false;
	threadsPerBlock.command = "-t";
	threadsPerBlock.description = "Threads Per Block used for cuda kernels. Default is: 128";
	threadsPerBlock.type = Setting::UNSIGNED;
	threadsPerBlock.uValue = 128;
	settingsMap[threadsPerBlock.name] = threadsPerBlock;

	Setting maxIterations;
	maxIterations.name = "maxIterations";
	maxIterations.isSet = false;
	maxIterations.command = "-mi";
	maxIterations.description = "Threads Per Block used for cuda kernels. Default is: 128";
	maxIterations.type = Setting::UNSIGNED;
	maxIterations.uValue = 128;
	settingsMap[maxIterations.name] = maxIterations;

	Setting svm;
	svm.name = "svm";
	svm.isSet = false;
	svm.command = "-svm";
	svm.description = "Type of SVM to use, 'p' for parallel, 's' for sequential. Default is: s";
	svm.type = Setting::STRING;
	svm.sValue = "s";
	settingsMap[svm.name] = svm;

	Setting dataSet;
	dataSet.name = "dataSet";
	dataSet.isSet = false;
	dataSet.command = "-d";
	dataSet.description = "DataSet to use, i | a[1-9] | w[1-8]";
	dataSet.type = Setting::STRING;
	dataSet.sValue = "i";
	settingsMap[dataSet.name] = dataSet;

	Setting gamma;
	gamma.name = "gamma";
	gamma.isSet = false;
	gamma.command = "-g";
	gamma.description = "Gamma value used for gaussian kernel, default varies by dataSet.";
	gamma.type = Setting::DOUBLE;
	gamma.dValue = 0.5;
	settingsMap[gamma.name] = gamma;

	Setting constraint;
	constraint.name = "constraint";
	constraint.isSet = false;
	constraint.command = "-c";
	constraint.description = "Constraint for softmargin. Default is 999";
	constraint.type = Setting::DOUBLE;
	constraint.dValue = 999;
	settingsMap[constraint.name] = constraint;

	Setting step;
	step.name = "step";
	step.isSet = false;
	step.command = "-st";
	step.description = "Size of first step is algorithm. Default is 1";
	step.type = Setting::DOUBLE;
	step.dValue = 1;
	settingsMap[step.name] = step;

	Setting precission;
	precission.name = "precision";
	precission.isSet = false;
	precission.command = "-p";
	precission.description = "Precision of double values. Default is 1e-10";
	precission.type = Setting::DOUBLE;
	precission.dValue = 1e-10;
	settingsMap[precission.name] = precission;

	Setting help;
	help.name = "help";
	help.isSet = false;
	help.command = "-h";
	help.description = "Shows options available.";
	help.type = Setting::HELP;
	settingsMap[help.name] = help;

	Setting log;
	log.name = "log";
	log.isSet = false;
	log.command = "-l";
	log.description = "Define log level, {a:all, r:results, e:only errors, n:none} Default is r";
	log.type = Setting::STRING;
	log.sValue = "r";
	settingsMap[log.name] = log;

	for (auto it = settingsMap.begin(); it != settingsMap.end(); ++it)
	{
		Setting *st = &(*it).second;
		auto arg = Utils::GetComandVariable(argc, argv, st->command);
		if (st->type == Setting::HELP)
			ShowHelp();
		else if (arg.empty())
		{
			continue;
		}
		else if (st->type == Setting::UNSIGNED && Utils::TryParseInt(arg, st->uValue))
		{
		}
		else if (st->type == Setting::STRING)
		{
			st->sValue = arg;
		}
		else if (st->type == Setting::DOUBLE && Utils::TryParseDouble(arg, st->dValue))
		{
		}
		else continue;
		st->isSet = true;
	}
}

void Settings::GetUnsigned(string key, unsigned& value)
{
	value = settingsMap[key].uValue;
}

string Settings::GetString(string key)
{
	auto s = settingsMap[key].sValue;
	return s;
}

void Settings::GetDouble(string key, double& value)
{
	value = settingsMap[key].dValue;
}

void Settings::GetDouble(string key, double& value, double def)
{
	if (settingsMap[key].isSet)
		value = settingsMap[key].dValue;
	else{
		value = def;
		settingsMap[key].dValue = def;
	}
}

Settings::~Settings()
{
}