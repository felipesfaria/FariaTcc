#include "stdafx.h"
#include "DataSet.h"
#include "Utils.h"
#include <sstream>
#include <fstream>
#include "Settings.h"
using namespace std;

DataSet::DataSet()
{
	auto dataSet = Settings::instance()->GetString("dataSet");
	FileName = GetFileName(dataSet);
	InitData(dataSet);
	ReadFile();
	Logger::instance()->Stats("FileName", FileName);
	Logger::instance()->Stats("Samples", nSamples);
	Logger::instance()->Stats("Features", nFeatures);
	Logger::instance()->Stats("Classes", nClasses);
	Settings::instance()->GetUnsigned("folds", nFolds);
	Utils::Shuffle(X, Y);
}

DataSet::~DataSet()
{
}

void DataSet::InitData(string arg)
{
	kernelType = KernelType::GAUSSIAN;
	switch (arg[0])
	{
	case 'a':
		C = 100;
		Gama = 0.5;
		break;
	case 'w':
		C = 64;
		Gama = 7.8125;
		break;
	case 'i':
	default:
		C = 4;
		Gama = 0.1;
	}
}

void DataSet::ReadFile()
{
	ifstream       file;
	file.open(FileName, ifstream::in);

	if (!file.good())
		throw(exception("Error: File not found"));

	while (readRow(file))
	{}
	nSamples = X.size();
	file.close();
}
bool DataSet::readRow(istream& str)
{
	string line;
	if (!getline(str, line))
		return false;

	stringstream lineStream(line);
	string cell;

	getline(lineStream, cell, ' ');
	double y;
	if (!Utils::TryParseDouble(cell, y))
		throw exception("Missing y value.");
	vector<double> x;
	int count = X.size();
	int i = 1;
	while (getline(lineStream, cell, ' '))
	{
		auto elems = Utils::split(cell, ':');
		int index = stoi(elems[0]);
		while (i++<index)
			x.push_back(0);
		x.push_back(stod(elems[1]));
	}
	if (nFeatures < x.size())
		nFeatures = x.size();
	X.push_back(x);
	Y.push_back(y);
	return true;
}

void DataSet::InitFoldSets(TrainingSet *ts, ValidationSet*vs, int fold)
{
	int vStart = nSamples*(fold - 1) / nFolds;
	int vEnd = nSamples*fold / nFolds;
	ts->Init(nSamples - (vEnd - vStart), nFeatures);
	vs->Init((vEnd - vStart), nFeatures);
	for (int i = 0; i < nSamples; i++)
	{
		if (i >= vStart&&i<vEnd)
			vs->PushSample(X[i], Y[i]);
		else
			ts->PushSample(X[i], Y[i]);
	}
}

string DataSet::GetFileName(string arg)
{
	auto argChar = arg.c_str();
	string fileName = "Data/";
	switch (argChar[0])
	{
	case 'a':
		fileName += "adult";
		break;
	case 'w':
		fileName += "web";
		break;
	case 'i':
	default:
		fileName += "iris";
		break;
	}
	if (arg.size() == 2 && argChar[0] == 'w' || argChar[0] == 'a')
	{
		fileName += argChar[1];
	}
	fileName += ".data";
	return fileName;
}
