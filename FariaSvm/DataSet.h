#pragma once
#include "Enums.h"
#include "ValidationSet.h"
#include "TrainingSet.h"
#include <vector>

using namespace std;

class DataSet
{
public:
	string FileName;
	int nSamples;
	int nFeatures = 0;
	int nClasses = 2;
	unsigned nFolds;
	int C;
	KernelType kernelType;
	double Gama=-1;
	vector<vector<double>> X;
	vector<double> Y;
	DataSet();
	~DataSet();
	void InitData(string arg="");
	void InitFoldSets(TrainingSet *ts, ValidationSet *vs, int fold);
	static string GetFileName(string arg);
private:
	vector<double> m_doubles;
	vector<long> m_longs;
	string classes[2];

	void ReadFile();

	bool readRow(istream& str);
};

