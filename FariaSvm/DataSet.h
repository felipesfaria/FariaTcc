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
	bool IsCsv;
	int nSamples;
	int nFeatures;
	int nClasses;
	unsigned nFolds;
	int C;
	KernelType kernelType;
	double Gama=-1;
	vector<vector<double>> X;
	vector<double> Y;
	std::string const& operator[](size_t index) const;
	size_t size() const;
	DataSet();
	~DataSet();
	void InitData(string arg="");
	void InitFoldSets(TrainingSet *ts, ValidationSet *vs, int fold);
private:
	vector<double> m_doubles;
	vector<long> m_longs;
	string classes[2];

	void ReadFile();
	bool readNextRow(istream& str);

	void InitAdult(char c);
	void InitAdult(int n);
	void InitWeb(char c);
	void InitWeb(int n);
	void InitIris();

	void readIndexedData(istream& str);
	void readCsvData(istream& str);
};

