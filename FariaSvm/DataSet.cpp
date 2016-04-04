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
	InitData(dataSet);
	
	Logger::instance()->Stats("FileName", FileName);
	Logger::instance()->Stats("Samples", nSamples);
	Logger::instance()->Stats("Features", nFeatures);
	Logger::instance()->Stats("Classes", nClasses);

	Settings::instance()->GetUnsigned("folds", nFolds);

	ReadFile();
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
		InitAdult(arg[1]);
		break;
	case 'w':
		InitWeb(arg[1]);
		break;
	case 'i':
	default:
		InitIris();
	}
}

void DataSet::InitAdult(char c)
{
	return InitAdult(c - '0');
}
void DataSet::InitAdult(int n)
{
	IsCsv = false;
	nFeatures = 123;
	nClasses = 2;
	C = 100;
	Gama = 0.5;
	switch (n)
	{
	case 1:
		FileName = "Data/adult1.data";
		nSamples = 1605;
		break;
	case 2:
		FileName = "Data/adult2.data";
		nSamples = 2265;
		break;
	case 3: 
		FileName = "Data/adult3.data";
		nSamples = 3185;
		break;
	case 4:
		FileName = "Data/adult4.data";
		nSamples = 4781;
		break;
	case 5:
		FileName = "Data/adult5.data";
		nSamples = 6414;
		break;
	case 6:
		FileName = "Data/adult6.data";
		nSamples = 11220;
		break;
	case 7: 
		FileName = "Data/adult7.data";
		nSamples = 16100;
		break;
	case 8: 
		FileName = "Data/adult8.data";
		nSamples = 22696;
		break;
	case 9: 
		FileName = "Data/adult9.data";
		nSamples = 32561;
		break;
	default:
		throw(exception("Invalid adult type"));
	}
}

void DataSet::InitWeb(char c)
{
	InitWeb(c - '0');
}
void DataSet::InitWeb(int n)
{
	IsCsv = false;
	nFeatures = 300;
	nClasses = 2;
	C = 64;
	Gama = 7.8125;
	switch (n)
	{
	case 1:
		FileName = "Data/web1.data";
		nSamples = 2477;
		break;
	case 2:
		FileName = "Data/web2.data";
		nSamples = 3470;
		break;
	case 3:
		FileName = "Data/web3.data";
		nSamples = 4912;
		break;
	case 4:
		FileName = "Data/web4.data";
		nSamples = 7366;
		break;
	case 5:
		FileName = "Data/web5.data";
		nSamples = 9888;
		break;
	case 6:
		FileName = "Data/web6.data";
		nSamples = 17188;
		break;
	case 7:
		FileName = "Data/web7.data";
		nSamples = 24692;
		break;
	case 8:
		FileName = "Data/web8.data";
		nSamples = 49749;
		break;
	default:
		throw(exception("Invalid Web type"));
	}
}

void DataSet::InitIris()
{
	FileName = "Data/iris.csv";
	IsCsv = true;
	nSamples = 100;
	nFeatures = 4;
	nClasses = 2;
	C = 16;
	Gama = 0.5;
}

bool DataSet::readNextRow(istream& str)
{
	if (IsCsv)
		readCsvData(str);
	else
		readIndexedData(str);
	if (str)
		return true;
	return false;
}

void DataSet::ReadFile()
{
	ifstream       file;
	file.open(FileName, ifstream::in);

	if (!file.good())
		throw(exception("Error: File not found"));

	for (int i = 0; i < nSamples; i++)
		readNextRow(file);
	file.close();
}
void DataSet::readIndexedData(istream& str)
{
	string line;
	getline(str, line);

	if (line.size() == 0) return;
	stringstream lineStream(line);
	string cell;

	getline(lineStream, cell, ' ');
	double y;
	if (!Utils::TryParseDouble(cell, y))
		throw exception("Missing y value.");
	vector<double> x;
	for (int i = 0; i < nFeatures; ++i)
		x.push_back(0);

	while (getline(lineStream, cell, ' '))
	{
		auto elems = Utils::split(cell, ':');
		int index = stoi(elems[0]);
		x[index - 1] = stod(elems[1]);
	}
	X.push_back(x);
	Y.push_back(y);
}

void DataSet::readCsvData(istream& str)
{
	string         line;
	getline(str, line);

	stringstream   lineStream(line);
	string         cell;

	vector<string> m_data;
	vector<double> x;
	while (getline(lineStream, cell, ','))
	{
		m_data.push_back(cell);
		try
		{
			double t;
			t = stod(cell);
			x.push_back(t);
			continue;
		}
		catch (invalid_argument&)
		{

		}

		try
		{
			long t;
			t = stol(cell);
			continue;
		}
		catch (invalid_argument&)
		{

		}
	}
	//Last line didn't read anything
	if (m_data.size() == 0) return;

	string myClass = m_data[m_data.size() - 1];

	//Find out classes
	//static string SvmData::classes[2];
	if (classes[1].empty())
		if (classes[0].empty())
			classes[0] = myClass;
		else if (classes[0].compare(myClass) != 0)
			classes[1] = myClass;
	double y;
	if (classes[0].compare(myClass) == 0)
		y = 1;
	else
		y = -1;
	X.push_back(x);
	Y.push_back(y);
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